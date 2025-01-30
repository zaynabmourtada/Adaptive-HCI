package com.developer27.xamera.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.withContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.core.MatOfPoint
import org.opencv.video.KalmanFilter
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.IValue
import java.util.LinkedList
import kotlin.math.min
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classId: Int
)

data class FrameData(
    val x: Double,
    val y: Double,
    val area: Double,
    val frameCount: Int
)

object Settings {
    object Model {
        const val inputSize = 960
        const val outputTensorStride = 6 // Number of values per detection
    }

    object Trace {
        var lineLimit = 50
        var splineStep = 0.01
        var originalLineColor = Scalar(255.0, 0.0, 0.0) // Red
        var splineLineColor = Scalar(0.0, 0.0, 255.0)  // Blue
        var lineThickness = 4
    }

    object BoundingBox {
        var boxColor = Scalar(0.0, 255.0, 0.0) // Green
        var boxThickness = 2
    }

    object Brightness {
        var factor = 2.0
        var threshold = 150.0
    }

    object Debug {
        var enableToasts = true
        var enableLogging = true
    }
}

class VideoProcessor(private val context: Context) {
    private var module: Module? = null

    // For line-drawing (visualization)
    private val rawDataList = LinkedList<Point>()
    private val smoothDataList = LinkedList<Point>()

    private var frameCount = 0

    // Storing final data
    private val preFilter4Ddata = mutableListOf<FrameData>()
    private val postFilter4Ddata = mutableListOf<FrameData>()

    init {
        initOpenCV()
        KalmanHelper.initKalmanFilter()
    }

    private fun initOpenCV() {
        try {
            System.loadLibrary("opencv_java4")
            Log.d("VideoProcessor", "OpenCV loaded successfully.")
        } catch (e: UnsatisfiedLinkError) {
            Log.e("VideoProcessor", "OpenCV failed to load: ${e.message}")
        }
    }

    fun setModel(module: Module) {
        this.module = module
        logCat("Model loaded successfully")
    }

    fun clearTrackingData() {
        frameCount = 0
        preFilter4Ddata.clear()
        postFilter4Ddata.clear()
        rawDataList.clear()
        smoothDataList.clear()
        showToast("Tracking data reset.")
    }

    fun getPostFilterData(): List<FrameData> {
        return postFilter4Ddata.toList()
    }

    fun processFrame(bitmap: Bitmap, callback: (Bitmap?) -> Unit) {
        CoroutineScope(Dispatchers.Default).launch {
            val result = try {
                processFrameInternalCONTOUR(bitmap)
            } catch (e: Exception) {
                logCat("Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) {
                callback(result) // Return result on the main thread
            }
        }
    }

    private fun processFrameInternalCONTOUR(bitmap: Bitmap): Bitmap? {
        val mat = Mat()
        val originalMat = Mat()
        val resizedMat = Mat()

        return try {
            // Convert bitmap to Mat using OpenCV's Utils.bitmapToMat
            Utils.bitmapToMat(bitmap, originalMat)

            // Preprocess the frame to enhance the light blobs
            val preprocessedMat = Preprocessing.preprocessFrame(originalMat)

            // Find contours in the preprocessed image
            val contours = ContourDetection.findContours(preprocessedMat)

            // Find the largest contour (blob of light)
            val largestContour = ContourDetection.findLargestContour(contours)

            // If a contour is found, draw it and calculate its center of mass
            if (largestContour != null) {
                // Draw the largest contour on the original image
                ContourDetection.drawContour(originalMat, largestContour)

                // Calculate the center of mass of the largest contour
                val center = ContourDetection.calculateCenterOfMass(largestContour)
                rawDataList.add(center)

                // Apply Kalman filter to the center point
                val (fx, fy) = KalmanHelper.applyKalmanFilter(center)
                smoothDataList.add(Point(fx, fy))

                // Keep the trace lines limited
                if (rawDataList.size > Settings.Trace.lineLimit) {
                    rawDataList.pollFirst()
                }
                if (smoothDataList.size > Settings.Trace.lineLimit) {
                    smoothDataList.pollFirst()
                }

                // Draw raw trace
                TraceRenderer.drawRawTrace(rawDataList, originalMat)

                // Draw smoothed trace
                TraceRenderer.drawSplineCurve(smoothDataList, originalMat)
            }

            // Convert Mat back to Bitmap using OpenCV's Utils.matToBitmap
            val outputBitmap = Bitmap.createBitmap(originalMat.cols(), originalMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(originalMat, outputBitmap)
            outputBitmap
        } catch (e: Exception) {
            logCat("Error processing frame: ${e.message}", e)
            null
        } finally {
            // Release Mats in the finally block to ensure they are always released
            mat.release()
            originalMat.release()
            resizedMat.release()
        }
    }

    private suspend fun processFrameInternalYOLO(bitmap: Bitmap): Bitmap? {
        val mat = Mat()
        val originalMat = Mat()
        val resizedMat = Mat()

        return try {
            // Convert bitmap to Mat using OpenCV's Utils.bitmapToMat
            Utils.bitmapToMat(bitmap, originalMat)

            // Resize and pad the image to 960x960
            YOLOHelper.resizeAndPad(originalMat, resizedMat, Settings.Model.inputSize, Settings.Model.inputSize)

            // Convert Mat to Tensor
            val inputTensor = ImageUtils.matToFloat32Tensor(resizedMat)

            // Run inference on the background thread
            val outputTensor = withContext(Dispatchers.Default) {
                module?.forward(IValue.from(inputTensor))?.toTensor()
            }

            // Process output tensor to get bounding boxes
            val boundingBoxes = outputTensor?.let { YOLOHelper.parseYOLOOutputTensor(it, originalMat.cols(), originalMat.rows()) } ?: emptyList()

            // Draw bounding boxes on the original image
            YOLOHelper.drawBoundingBoxes(originalMat, boundingBoxes)

            // Calculate the center of each bounding box and add to rawDataList
            for (box in boundingBoxes) {
                val center = YOLOHelper.calculateCenter(box)
                rawDataList.add(center)

                // Apply Kalman filter to the center point
                val (fx, fy) = KalmanHelper.applyKalmanFilter(center)
                smoothDataList.add(Point(fx, fy))
            }

            // Keep the trace lines limited
            if (rawDataList.size > Settings.Trace.lineLimit) {
                rawDataList.pollFirst()
            }
            if (smoothDataList.size > Settings.Trace.lineLimit) {
                smoothDataList.pollFirst()
            }

            // Draw raw trace
            TraceRenderer.drawRawTrace(rawDataList, originalMat)

            // Draw smoothed trace
            TraceRenderer.drawSplineCurve(smoothDataList, originalMat)

            // Convert Mat back to Bitmap using OpenCV's Utils.matToBitmap
            val outputBitmap = Bitmap.createBitmap(originalMat.cols(), originalMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(originalMat, outputBitmap)
            outputBitmap
        } catch (e: Exception) {
            logCat("Error processing frame: ${e.message}", e)
            null
        } finally {
            // Release Mats in the finally block to ensure they are always released
            mat.release()
            originalMat.release()
            resizedMat.release()
        }
    }

    private fun showToast(msg: String) {
        if (Settings.Debug.enableToasts) {
            Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
        }
    }

    private fun logCat(message: String, throwable: Throwable? = null) {
        if (Settings.Debug.enableLogging) {
            if (throwable != null) {
                Log.e("VideoProcessor", message, throwable)
            } else {
                Log.d("VideoProcessor", message)
            }
        }
    }
}

object TraceRenderer {
    fun drawRawTrace(data: List<Point>, image: Mat) {
        for (i in 1 until data.size) {
            Imgproc.line(
                image,
                data[i - 1],
                data[i],
                Settings.Trace.originalLineColor,
                Settings.Trace.lineThickness
            )
        }
    }

    fun drawSplineCurve(data: List<Point>, image: Mat) {
        val splinePair = SplineHelper.applySplineInterpolation(data) ?: return
        val (splineX, splineY) = splinePair

        var prevPoint: Point? = null
        var t = 0.0
        val maxT = (data.size - 1).toDouble()
        while (t <= maxT) {
            val currentPoint = Point(splineX.value(t), splineY.value(t))
            prevPoint?.let {
                Imgproc.line(
                    image,
                    it,
                    currentPoint,
                    Settings.Trace.splineLineColor,
                    Settings.Trace.lineThickness
                )
            }
            prevPoint = currentPoint
            t += Settings.Trace.splineStep
        }
    }
}

object SplineHelper {
    fun applySplineInterpolation(data: List<Point>):
            Pair<org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction,
                    org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction>? {

        if (data.size < 2) return null
        val interpolator = SplineInterpolator()
        val xData = data.map { it.x }.toDoubleArray()
        val yData = data.map { it.y }.toDoubleArray()
        val tData = data.indices.map { it.toDouble() }.toDoubleArray()

        val splineX = interpolator.interpolate(tData, xData)
        val splineY = interpolator.interpolate(tData, yData)
        return splineX to splineY
    }
}

object KalmanHelper {
    private lateinit var kalmanFilter: KalmanFilter

    fun initKalmanFilter() {
        kalmanFilter = KalmanFilter(4, 2)

        kalmanFilter._transitionMatrix = Mat.eye(4, 4, CvType.CV_32F).apply {
            put(0, 2, 1.0)
            put(1, 3, 1.0)
        }
        kalmanFilter._measurementMatrix = Mat.eye(2, 4, CvType.CV_32F)
        kalmanFilter._processNoiseCov = Mat.eye(4, 4, CvType.CV_32F).apply { setTo(Scalar(1e-4)) }
        kalmanFilter._measurementNoiseCov = Mat.eye(2, 2, CvType.CV_32F).apply { setTo(Scalar(1e-2)) }
        kalmanFilter._errorCovPost = Mat.eye(4, 4, CvType.CV_32F)
    }

    fun applyKalmanFilter(point: Point): Pair<Double, Double> {
        val measurement = Mat(2, 1, CvType.CV_32F).apply {
            put(0, 0, point.x)
            put(1, 0, point.y)
        }

        kalmanFilter.predict()
        val corrected = kalmanFilter.correct(measurement)

        val fx = corrected[0, 0][0]
        val fy = corrected[1, 0][0]

        return fx to fy
    }
}

object Preprocessing {
    fun preprocessFrame(src: Mat): Mat {
        val grayMat = Preprocessing.applyGrayscale(src)
        val enhancedMat = Preprocessing.enhanceBrightness(grayMat)
        grayMat.release()

        val thresholdMat = Preprocessing.conditionalThresholding(enhancedMat)
        enhancedMat.release()

        val blurredMat = Preprocessing.applyGaussianBlur(thresholdMat)
        thresholdMat.release()

        val closedMat = Preprocessing.applyMorphologicalClosing(blurredMat)
        blurredMat.release()
        return closedMat
    }

    fun applyGrayscale(frame: Mat): Mat {
        val grayMat = Mat()
        Imgproc.cvtColor(frame, grayMat, Imgproc.COLOR_BGR2GRAY)
        return grayMat
    }
    fun enhanceBrightness(image: Mat): Mat = Mat().apply {
        Core.multiply(image, Scalar(Settings.Brightness.factor), this)
    }
    fun conditionalThresholding(image: Mat): Mat {
        val thresholdMat = Mat()
        Imgproc.threshold(
            image, thresholdMat,
            Settings.Brightness.threshold,
            255.0,
            Imgproc.THRESH_TOZERO
        )
        return thresholdMat
    }
    fun applyGaussianBlur(image: Mat): Mat {
        val blurredMat = Mat()
        Imgproc.GaussianBlur(image, blurredMat, Size(5.0, 5.0), 0.0)
        return blurredMat
    }
    fun applyMorphologicalClosing(image: Mat): Mat {
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val closedImage = Mat()
        Imgproc.morphologyEx(image, closedImage, Imgproc.MORPH_CLOSE, kernel)
        return closedImage
    }
}

object ImageUtils {
    fun matToFloat32Tensor(mat: Mat): Tensor {
        // Ensure the Mat is in the correct format (3-channel RGB)
        require(mat.channels() == 3) { "Input Mat must have 3 channels (RGB)" }

        // Allocate a direct ByteBuffer with the correct capacity
        val inputBuffer = ByteBuffer.allocateDirect(mat.cols() * mat.rows() * 3) // 1 byte per channel
        inputBuffer.order(ByteOrder.nativeOrder())

        // Copy pixel data from the Mat to the ByteBuffer
        for (row in 0 until mat.rows()) {
            for (col in 0 until mat.cols()) {
                val pixel = mat.get(row, col)
                inputBuffer.put((pixel[0].toInt() and 0xFF).toByte()) // B
                inputBuffer.put((pixel[1].toInt() and 0xFF).toByte()) // G
                inputBuffer.put((pixel[2].toInt() and 0xFF).toByte()) // R
            }
        }

        // Log tensor shape and buffer size
        Log.d("VideoProcessor", "Tensor shape: [1, 3, ${mat.rows()}, ${mat.cols()}]")
        Log.d("VideoProcessor", "ByteBuffer size: ${inputBuffer.capacity()}")

        // Create a PyTorch Tensor from the ByteBuffer
        return Tensor.fromBlob(inputBuffer, longArrayOf(1, 3, mat.rows().toLong(), mat.cols().toLong()))
    }
}

object ContourDetection {
    fun findContours(mat: Mat): List<MatOfPoint> {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()
        return contours
    }

    fun findLargestContour(contours: List<MatOfPoint>): MatOfPoint? {
        return contours.maxByOrNull { Imgproc.contourArea(it) }
    }

    fun drawContour(mat: Mat, contour: MatOfPoint) {
        Imgproc.drawContours(mat, listOf(contour), -1, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)
    }

    fun calculateCenterOfMass(contour: MatOfPoint): Point {
        val moments = Imgproc.moments(contour)
        val centerX = moments.m10 / moments.m00
        val centerY = moments.m01 / moments.m00
        return Point(centerX, centerY)
    }
}

object YOLOHelper {
    fun resizeAndPad(src: Mat, dst: Mat, targetWidth: Int, targetHeight: Int) {
        val scale = min(targetWidth.toDouble() / src.cols(), targetHeight.toDouble() / src.rows())
        val resizedWidth = (src.cols() * scale).toInt()
        val resizedHeight = (src.rows() * scale).toInt()

        Imgproc.resize(src, dst, Size(resizedWidth.toDouble(), resizedHeight.toDouble()))

        val paddedMat = Mat(targetHeight, targetWidth, CvType.CV_8UC3, Scalar(0.0, 0.0, 0.0))

        val xOffset = (targetWidth - resizedWidth) / 2
        val yOffset = (targetHeight - resizedHeight) / 2

        dst.copyTo(paddedMat.submat(yOffset, yOffset + resizedHeight, xOffset, xOffset + resizedWidth))
        paddedMat.copyTo(dst)
        paddedMat.release()
    }

    fun parseYOLOOutputTensor(outputTensor: Tensor, originalWidth: Int, originalHeight: Int): List<BoundingBox> {
        val boundingBoxes = mutableListOf<BoundingBox>()
        val outputArray = outputTensor.dataAsFloatArray
        val numDetections = outputTensor.shape()[1].toInt()

        for (i in 0 until numDetections) {
            val x1 = outputArray[i * Settings.Model.outputTensorStride + 0]
            val y1 = outputArray[i * Settings.Model.outputTensorStride + 1]
            val x2 = outputArray[i * Settings.Model.outputTensorStride + 2]
            val y2 = outputArray[i * Settings.Model.outputTensorStride + 3]
            val confidence = outputArray[i * Settings.Model.outputTensorStride + 4]
            val classId = outputArray[i * Settings.Model.outputTensorStride + 5].toInt()

            val scaleX = originalWidth.toFloat() / Settings.Model.inputSize
            val scaleY = originalHeight.toFloat() / Settings.Model.inputSize
            val scaledX1 = x1 * scaleX
            val scaledY1 = y1 * scaleY
            val scaledX2 = x2 * scaleX
            val scaledY2 = y2 * scaleY

            boundingBoxes.add(BoundingBox(scaledX1, scaledY1, scaledX2, scaledY2, confidence, classId))
        }
        return boundingBoxes
    }

    fun drawBoundingBoxes(mat: Mat, boundingBoxes: List<BoundingBox>) {
        for (box in boundingBoxes) {
            val topLeft = Point(box.x1.toDouble(), box.y1.toDouble())
            val bottomRight = Point(box.x2.toDouble(), box.y2.toDouble())

            Imgproc.rectangle(mat, topLeft, bottomRight, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)

            val label = "User_1 (${"%.2f".format(box.confidence * 100)}%)"
            val fontScale = 0.6
            val thickness = 1
            val baseline = IntArray(1)
            val textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, thickness, baseline)

            val textX = (box.x1).toInt()
            val textY = (box.y1 - 5).toInt().coerceAtLeast(10)

            Imgproc.rectangle(
                mat,
                Point(textX.toDouble(), textY.toDouble() + baseline[0]),
                Point(textX + textSize.width, textY - textSize.height),
                Settings.BoundingBox.boxColor,
                Imgproc.FILLED
            )

            Imgproc.putText(
                mat,
                label,
                Point(textX.toDouble(), textY.toDouble()),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                fontScale,
                Scalar(255.0, 255.0, 255.0),
                thickness
            )
        }
    }

    fun calculateCenter(box: BoundingBox): Point {
        val centerX = (box.x1 + box.x2) / 2.0
        val centerY = (box.y1 + box.y2) / 2.0
        return Point(centerX, centerY)
    }
}
