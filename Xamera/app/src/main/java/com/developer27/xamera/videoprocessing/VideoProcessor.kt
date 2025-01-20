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
import org.opencv.video.KalmanFilter
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.IValue
import java.util.LinkedList
import kotlin.math.min
import java.nio.ByteBuffer
import java.nio.ByteOrder

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
    private lateinit var kalmanFilter: KalmanFilter
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
        initKalmanFilter()
    }

    private fun initOpenCV() {
        if (OpenCVLoader.initDebug()) {
            showToast("OpenCV loaded successfully")
        } else {
            Log.e("VideoProcessor", "OpenCV failed to load.")
        }
    }

    private fun initKalmanFilter() {
        kalmanFilter = KalmanFilter(4, 2)
        kalmanFilter._transitionMatrix = Mat.eye(4, 4, CvType.CV_32F).apply {
            put(0, 2, 1.0)
            put(1, 3, 1.0)
        }
        kalmanFilter._measurementMatrix = Mat.eye(2, 4, CvType.CV_32F)
        kalmanFilter._processNoiseCov = Mat.eye(4, 4, CvType.CV_32F).apply {
            setTo(Scalar(1e-4))
        }
        kalmanFilter._measurementNoiseCov = Mat.eye(2, 2, CvType.CV_32F).apply {
            setTo(Scalar(1e-2))
        }
        kalmanFilter._errorCovPost = Mat.eye(4, 4, CvType.CV_32F)
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
                processFrameInternal(bitmap)
            } catch (e: Exception) {
                logCat("Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) {
                callback(result) // Return result on the main thread
            }
        }
    }

    private suspend fun processFrameInternal(bitmap: Bitmap): Bitmap? {
        val mat = Mat()
        val originalMat = Mat()
        val resizedMat = Mat()

        return try {
            logCat("Starting frame processing")

            // Convert bitmap to Mat using OpenCV's Utils.bitmapToMat
            logCat("Converting bitmap to Mat")
            Utils.bitmapToMat(bitmap, originalMat)
            logCat("Bitmap converted to Mat: ${originalMat.cols()}x${originalMat.rows()}")

            // Resize and pad the image to 960x960
            logCat("Resizing and padding image")
            resizeAndPad(originalMat, resizedMat, Settings.Model.inputSize, Settings.Model.inputSize)
            logCat("Image resized and padded: ${resizedMat.cols()}x${resizedMat.rows()}")

            // Convert Mat to Tensor
            logCat("Converting Mat to Tensor")
            val inputTensor = ImageUtils.matToFloat32Tensor(resizedMat)
            logCat("Tensor created with shape: ${inputTensor.shape().joinToString()}")

            // Run inference on the background thread
            logCat("Running inference")
            val outputTensor = withContext(Dispatchers.Default) {
                module?.forward(IValue.from(inputTensor))?.toTensor()
            }
            logCat("Inference completed successfully")

            // Process output tensor to get bounding boxes
            logCat("Processing output tensor")
            val boundingBoxes = outputTensor?.let { parseYOLOOutputTensor(it, originalMat.cols(), originalMat.rows()) } ?: emptyList()
            logCat("Bounding boxes detected: ${boundingBoxes.size}")

            // Draw bounding boxes on the original image
            logCat("Drawing bounding boxes")
            drawBoundingBoxes(originalMat, boundingBoxes)

            // Calculate the center of each bounding box and add to rawDataList
            logCat("Calculating centers and applying Kalman filter")
            for (box in boundingBoxes) {
                val center = calculateCenter(box)
                rawDataList.add(center)

                // Apply Kalman filter to the center point
                val (fx, fy) = applyKalmanFilter(center)
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
            logCat("Drawing raw trace")
            TraceRenderer.drawRawTrace(rawDataList, originalMat)

            // Draw smoothed trace
            logCat("Drawing smoothed trace")
            TraceRenderer.drawSplineCurve(smoothDataList, originalMat)

            // Convert Mat back to Bitmap using OpenCV's Utils.matToBitmap
            logCat("Converting Mat back to Bitmap")
            val outputBitmap = Bitmap.createBitmap(originalMat.cols(), originalMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(originalMat, outputBitmap)
            logCat("Frame processing completed successfully")
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

    private fun resizeAndPad(src: Mat, dst: Mat, targetWidth: Int, targetHeight: Int) {
        // Calculate the scale to maintain aspect ratio
        val scale = min(targetWidth.toDouble() / src.cols(), targetHeight.toDouble() / src.rows())
        val resizedWidth = (src.cols() * scale).toInt()
        val resizedHeight = (src.rows() * scale).toInt()

        // Resize the image to fit within the target dimensions while maintaining aspect ratio
        Imgproc.resize(src, dst, Size(resizedWidth.toDouble(), resizedHeight.toDouble()))

        // Create a square padded image with black borders
        val paddedMat = Mat(targetHeight, targetWidth, CvType.CV_8UC3, Scalar(0.0, 0.0, 0.0)) // Black padding

        // Calculate offsets to center the resized image in the padded square
        val xOffset = (targetWidth - resizedWidth) / 2
        val yOffset = (targetHeight - resizedHeight) / 2

        // Copy the resized image into the center of the padded square
        dst.copyTo(paddedMat.submat(yOffset, yOffset + resizedHeight, xOffset, xOffset + resizedWidth))

        // Copy the padded result back to the destination Mat
        paddedMat.copyTo(dst)
        paddedMat.release()
    }

    private fun parseYOLOOutputTensor(outputTensor: Tensor, originalWidth: Int, originalHeight: Int): List<BoundingBox> {
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

            // Scale bounding box coordinates back to the original image size
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

    private fun drawBoundingBoxes(mat: Mat, boundingBoxes: List<BoundingBox>) {
        for (box in boundingBoxes) {
            val topLeft = Point(box.x1.toDouble(), box.y1.toDouble())
            val bottomRight = Point(box.x2.toDouble(), box.y2.toDouble())

            // Draw the bounding box
            Imgproc.rectangle(mat, topLeft, bottomRight, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)

            // Prepare text to display (class name and confidence score)
            val label = "User_1 (${"%.2f".format(box.confidence * 100)}%)"
            val fontScale = 0.6
            val thickness = 1
            val baseline = IntArray(1)
            val textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, thickness, baseline)

            // Calculate text position (above the bounding box)
            val textX = (box.x1).toInt()
            val textY = (box.y1 - 5).toInt().coerceAtLeast(10) // Ensure text doesn't go off the top of the image

            // Draw a background rectangle for the text
            Imgproc.rectangle(
                mat,
                Point(textX.toDouble(), textY.toDouble() + baseline[0]),
                Point(textX + textSize.width, textY - textSize.height),
                Settings.BoundingBox.boxColor,
                Imgproc.FILLED
            )

            // Draw the text
            Imgproc.putText(
                mat,
                label,
                Point(textX.toDouble(), textY.toDouble()),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                fontScale,
                Scalar(255.0, 255.0, 255.0), // White text
                thickness
            )
        }
    }

    private fun applyKalmanFilter(point: Point): Pair<Double, Double> {
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

    private fun preprocessFrame(src: Mat): Mat {
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

    private fun calculateCenter(box: BoundingBox): Point {
        val centerX = (box.x1 + box.x2) / 2.0
        val centerY = (box.y1 + box.y2) / 2.0
        return Point(centerX, centerY)
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

object Preprocessing {
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