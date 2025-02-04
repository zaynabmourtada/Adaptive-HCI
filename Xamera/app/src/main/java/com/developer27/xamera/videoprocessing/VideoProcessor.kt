package com.developer27.xamera.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Environment
import android.util.Log
import android.widget.Toast
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.video.KalmanFilter
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.util.LinkedList
import kotlin.math.exp


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

    // Switch Between YOLO vs Contour Detection
    fun processFrame(bitmap: Bitmap, callback: (Bitmap?) -> Unit) {
        CoroutineScope(Dispatchers.Default).launch {
            val result = try {
                // Switch Between YOLO vs Contour Detection
                //processFrameInternalYOLO(bitmap)
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
                // ContourDetection.drawContour(originalMat, largestContour)

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

    // add padding or resizing functions, switch to live stream from camera, add trace lines etc
    // building blocks availible
    private suspend fun processFrameInternalYOLO(bitmap: Bitmap): Bitmap? {
        return withContext(Dispatchers.IO) {
            try {
                val newBitMap = makeSquareAndResize(bitmap)
                // Convert Frame Bitmap to Tensor (scale pixel values to [0, 1])
                val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                    newBitMap,
                    floatArrayOf(0f, 0f, 0f),  // No mean subtraction
                    floatArrayOf(1f, 1f, 1f)   // No std division (scale to [0, 1])
                )

                // The tensor shape will be [1, 3, 960, 960] after unsqueeze
                val batchedInputTensor = Tensor.fromBlob(
                    inputTensor.dataAsFloatArray, // Original tensor data
                    longArrayOf(1, 3, 960, 960)  // New shape with batch dimension
                )

                // Ensure Model is Loaded
                if (module == null) {
                    Log.e("YOLOTest", "Model is NULL! Cannot run inference.")
                    return@withContext null
                }

                // Run YOLO Inference
                val outputTensor = module?.forward(IValue.from(batchedInputTensor))?.toTensor()
                if (outputTensor == null) {
                    Log.e("YOLOTest", "YOLO Inference failed! Output tensor is NULL.")
                    return@withContext null
                }
                Log.d("YOLOTest", "Full YOLO Output Tensor Shape: ${outputTensor.shape().contentToString()}")

                // Convert Bitmap to OpenCV Mat
                val originalMat = Mat()
                Utils.bitmapToMat(bitmap, originalMat)

                // Process YOLO Output & Draw Bounding Boxes
                val (boundingBoxes, listOfPoints) = YOLOHelper.parseYOLOOutputTensor(outputTensor, originalMat.cols(), originalMat.rows())
                YOLOHelper.drawBoundingBoxes(originalMat, boundingBoxes, listOfPoints)

                // Convert Mat back to Bitmap using OpenCV's Utils.matToBitmap
                val outputBitmap = Bitmap.createBitmap(originalMat.cols(), originalMat.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(originalMat, outputBitmap)
                originalMat.release()

                outputBitmap
            } catch (e: Exception) {
                Log.e("YOLOTest", "Error during inference: ${e.message}", e)
                null
            }
        }
    }

    private fun makeSquareAndResize(bitmap: Bitmap): Bitmap {
        // Convert Bitmap to OpenCV Mat
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        // Get original dimensions
        val height = mat.rows()
        val width = mat.cols()
        val maxDim = maxOf(height, width)

        // Calculate padding (black padding with zero pixels)
        val top = (maxDim - height) / 2
        val bottom = maxDim - height - top
        val left = (maxDim - width) / 2
        val right = maxDim - width - left

        // Create a padded Mat with black (zero) padding
        val paddedMat = Mat()
        Core.copyMakeBorder(mat, paddedMat, top, bottom, left, right, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0))

        // Resize to 960x960
        val resizedMat = Mat()
        Imgproc.resize(paddedMat, resizedMat, Size(960.0, 960.0), 0.0, 0.0, Imgproc.INTER_AREA)

        // Convert back to Bitmap
        val outputBitmap = Bitmap.createBitmap(960, 960, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(resizedMat, outputBitmap)

        // Release Mat memory
        mat.release()
        paddedMat.release()
        resizedMat.release()

        return outputBitmap
    }

    fun testYOLOsingleImage(context: Context) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                // Load image from assets & convert from stream to BitMap
                val assetManager = context.assets
                val inputStream = assetManager.open("test_frame.jpg") // Test image file
                val bitmap = BitmapFactory.decodeStream(inputStream)
                if (bitmap == null) {
                    Log.e("YOLOTest", "Bitmap is null! Image decoding failed.")
                    return@launch
                }

                // Convert Image BitMap to Tensor (scale pixel values to [0, 1])
                val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                    bitmap,
                    floatArrayOf(0f, 0f, 0f),  // No mean subtraction
                    floatArrayOf(1f, 1f, 1f)   // No std division (scale to [0, 1])
                )

                // The tensor shape will be [1, 3, 960, 960] after unsqueeze
                val batchedInputTensor = Tensor.fromBlob(
                    inputTensor.dataAsFloatArray, // Original tensor data
                    longArrayOf(1, 3, 960, 960)  // New shape with batch dimension
                )

                // Ensure Model is Loaded
                if (module == null) {
                    Log.e("YOLOTest", "Model is NULL! Cannot run inference.")
                    return@launch
                }

                // run YOLO Inference
                val outputTensor = module?.forward(IValue.from(batchedInputTensor))?.toTensor()
                if (outputTensor == null) {
                    Log.e("YOLOTest", "YOLO Inference failed! Output tensor is NULL.")
                    return@launch
                }
                Log.d("YOLOTest", "Full YOLO Output Tensor Shape: ${outputTensor.shape().contentToString()}")

                // Convert Bitmap to OpenCV Mat
                val originalMat = Mat()
                Utils.bitmapToMat(bitmap, originalMat)

                // Process YOLO Output & Draw Bounding Boxes
                val (boundingBoxes, listOfPoints) = YOLOHelper.parseYOLOOutputTensor(outputTensor, originalMat.cols(), originalMat.rows())
                YOLOHelper.drawBoundingBoxes(originalMat, boundingBoxes, listOfPoints)

                originalMat.release()

            } catch (e: Exception) {
                Log.e("YOLOTest", "Error during inference: ${e.message}", e)
            }
        }
    }

    private fun saveInferenceResult(context: Context, mat: Mat) {
        try {
            // Convert Mat back to Bitmap
            val outputBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(mat, outputBitmap)

            // Save in the public Downloads folder
            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val outputFile = File(downloadsDir, "yolo_inference_result.jpg")

            // Write the image file
            FileOutputStream(outputFile).use { fos ->
                outputBitmap.compress(Bitmap.CompressFormat.JPEG, 90, fos)
                fos.flush()
            }

            Log.d("YOLOTest", "Saved inference result at: ${outputFile.absolutePath}")

            // Release Mat resources
            mat.release()

        } catch (e: Exception) {
            Log.e("YOLOTest", "Failed to save image: ${e.message}", e)
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
    fun parseYOLOOutputTensor(outputTensor: Tensor, originalWidth: Int, originalHeight: Int): Pair<List<BoundingBox>, List<Point>> {
        val boundingBoxes = mutableListOf<BoundingBox>()
        val listOfPoints = mutableListOf<Point>()
        val outputArray = outputTensor.dataAsFloatArray
        val numDetections = outputTensor.shape()[2].toInt() // 18900
        Log.d("YOLOTest", "Total detected objects: $numDetections")

        var bestD: Int = 0
        var bestX: Float = 0f
        var bestY: Float = 0f
        var bestW: Float = 0f
        var bestH: Float = 0f
        var bestC: Float = 0f

        for (i in 0 until numDetections) {
            val x_center = outputArray[i]       // First column (x_center)
            val y_center = outputArray[i + (numDetections * 1)]   // Second column (y_center)
            val width = outputArray[i + (numDetections * 2)]      // Third column (width)
            val height = outputArray[i + (numDetections * 3)]     // Fourth column (height)
            val confidence = outputArray[i + (numDetections * 4)] // Fifth column (confidence)
            //Log.d("YOLOTest", "DETECTION $i: confidence=${String.format("%.8f", confidence)}, x_center=$x_center, y_center=$y_center, width=$width, height=$height")
            if (confidence > bestC) {
                bestD = i
                bestX = x_center
                bestY = y_center
                bestW = width
                bestH = height
                bestC = confidence
            }
        }
        Log.d("YOLOTest", "BEST DETECTION $bestD: confidence=${String.format("%.8f", bestC)}, x_center=$bestX, y_center=$bestY, width=$bestW, height=$bestH")
        // Convert from (center_x, center_y, width, height) to (x1, y1, x2, y2)
        val x1 = bestX - (bestW / 2)
        val y1 = bestY - (bestH / 2)
        val x2 = bestX + (bestW / 2)
        val y2 = bestY + (bestH / 2)
        //Log.d("YOLOTest", "BOUND BOX: confidence=${String.format("%.8f", bestC)}, X1=$x1: Y1=$y1, X2=$x2, Y2=$y2")
        // Scale bounding boxes back to original image size
        val scaleX = originalWidth.toFloat() / 960f
        val scaleY = originalHeight.toFloat() / 960f
        val scaledX1 = x1 * scaleX
        val scaledY1 = y1 * scaleY
        val scaledX2 = x2 * scaleX
        val scaledY2 = y2 * scaleY
        //Log.d("YOLOTest", "BOUND BOX: confidence=${String.format("%.8f", bestC)}, scaledX1=$scaledX1: scaledY1=$scaledY1, scaledX2=$scaledX2, scaledY2=$scaledY2")
        // Add bounding Box
        boundingBoxes.add(BoundingBox(scaledX1, scaledY1, scaledX2, scaledY2, bestC, 1))
        listOfPoints.add(Point(bestX.toDouble(), bestY.toDouble()))
        return Pair(boundingBoxes, listOfPoints)
    }

    fun drawBoundingBoxes(mat: Mat, boundingBoxes: List<BoundingBox>, listOfPoints: List<Point>) {
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
}