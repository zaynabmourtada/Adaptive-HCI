@file:Suppress("SameParameterValue")

package com.developer27.xamera.videoprocessing

import android.content.Context
import android.graphics.Bitmap
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
import org.tensorflow.lite.DataType
import java.util.LinkedList
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage

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
    private var tfliteInterpreter: Interpreter? = null

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

    fun setTFLiteModel(model: Interpreter) {
        synchronized(this) { // 🔹 Ensure Thread Safety
            tfliteInterpreter = model
        }
        logCat("✅ TFLite Model set in VideoProcessor successfully!")
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
                processFrameInternalCONTOUR(bitmap)
                //processFrameInternalYOLO(bitmap)
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
        val originalMat = Mat()
        val preprocessedMat: Mat

        return try {
            // Convert bitmap to Mat using OpenCV
            Utils.bitmapToMat(bitmap, originalMat)
            // Preprocess the frame (Enhance light blobs, noise reduction, etc.)
            preprocessedMat = Preprocessing.preprocessFrame(originalMat)
            // Perform contour detection and get the center of mass
            val center = ContourDetection.processContourDetection(preprocessedMat)

            if (center != null) {
                rawDataList.add(center)

                // Apply Kalman filter to smooth tracking
                val (fx, fy) = KalmanHelper.applyKalmanFilter(center)
                smoothDataList.add(Point(fx, fy))

                // Maintain trace history within limits
                if (rawDataList.size > Settings.Trace.lineLimit) rawDataList.pollFirst()
                if (smoothDataList.size > Settings.Trace.lineLimit) smoothDataList.pollFirst()

                // Draw raw trace and smoothed trace directly on preprocessedMat
                //TraceRenderer.drawRawTrace(rawDataList, preprocessedMat)
                TraceRenderer.drawSplineCurve(smoothDataList, preprocessedMat)
            }

            // Convert preprocessedMat back to Bitmap (Returning the processed frame)
            val outputBitmap = Bitmap.createBitmap(preprocessedMat.cols(), preprocessedMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(preprocessedMat, outputBitmap)
            outputBitmap
        } catch (e: Exception) {
            logCat("Error processing frame: ${e.message}", e)
            null
        } finally {
            // Release Mats to free memory
            originalMat.release()
        }
    }

    // add padding or resizing functions, switch to live stream from camera, add trace lines etc
    // building blocks available
    private suspend fun processFrameInternalYOLO(bitmap: Bitmap): Bitmap? {
        return withContext(Dispatchers.IO) {
            try {
                val originalWidth = bitmap.width
                val originalHeight = bitmap.height
                val modelInputSize = 416 // Ensure this matches your model's expected size

                // Resize bitmap to YOLO model input size
                val resizedBitmap = Bitmap.createScaledBitmap(bitmap, modelInputSize, modelInputSize, true)

                // Convert Bitmap to TensorImage directly
                val tensorImage = TensorImage(DataType.FLOAT32)
                tensorImage.load(resizedBitmap)

                // Ensure TFLite Model is Loaded
                if (tfliteInterpreter == null) {
                    Log.e("YOLOTest", "TFLite Model is NULL! Cannot run inference.")
                    return@withContext null
                }

                // Prepare Output Tensor for TFLite
                val outputShape = arrayOf(1, 5, 3549) // Adjust based on your model's output
                val outputArray = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }

                // Run YOLO Inference
                tfliteInterpreter?.run(tensorImage.buffer, outputArray)

                Log.d("YOLOTest", "TFLite Inference Completed.")

                // Convert Bitmap to OpenCV Mat for Bounding Box Drawing
                val originalMat = Mat()
                Utils.bitmapToMat(bitmap, originalMat)

                // Process YOLO Output & Draw Bounding Boxes
                val (boundingBoxes, listOfPoints) = YOLOHelper.parseTFLiteOutputTensor(outputArray, originalWidth, originalHeight)
                YOLOHelper.drawBoundingBoxes(originalMat, boundingBoxes, listOfPoints)

                // Convert Mat back to Bitmap (Directly using original dimensions)
                val outputBitmap = Bitmap.createBitmap(originalWidth, originalHeight, Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(originalMat, outputBitmap)
                originalMat.release() // Free memory

                outputBitmap
            } catch (e: Exception) {
                Log.e("YOLOTest", "Error during inference: ${e.message}", e)
                null
            }
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
    fun applySplineInterpolation(data: List<Point>): Pair<org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction, org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction>? {
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
        val grayMat = applyGrayscale(src)
        val enhancedMat = enhanceBrightness(grayMat)
        grayMat.release()

        val thresholdMat = conditionalThresholding(enhancedMat)
        enhancedMat.release()

        val blurredMat = applyGaussianBlur(thresholdMat)
        thresholdMat.release()

        val closedMat = applyMorphologicalClosing(blurredMat)
        blurredMat.release()
        return closedMat
    }
    private fun applyGrayscale(frame: Mat): Mat {
        val grayMat = Mat()
        Imgproc.cvtColor(frame, grayMat, Imgproc.COLOR_BGR2GRAY)
        return grayMat
    }
    private fun enhanceBrightness(image: Mat): Mat = Mat().apply {
        Core.multiply(image, Scalar(Settings.Brightness.factor), this)
    }
    private fun conditionalThresholding(image: Mat): Mat {
        val thresholdMat = Mat()
        Imgproc.threshold(
            image, thresholdMat,
            Settings.Brightness.threshold,
            255.0,
            Imgproc.THRESH_TOZERO
        )
        return thresholdMat
    }
    private fun applyGaussianBlur(image: Mat): Mat {
        val blurredMat = Mat()
        Imgproc.GaussianBlur(image, blurredMat, Size(5.0, 5.0), 0.0)
        return blurredMat
    }
    private fun applyMorphologicalClosing(image: Mat): Mat {
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        val closedImage = Mat()
        Imgproc.morphologyEx(image, closedImage, Imgproc.MORPH_CLOSE, kernel)
        return closedImage
    }
}

object ContourDetection {
    fun processContourDetection(mat: Mat): Point? {
        // Find contours in the preprocessed image
        val contours = findContours(mat)

        // Find the largest contour (blob of light)
        val largestContour = findLargestContour(contours)

        return largestContour?.let {
            // Calculate the center of mass of the largest contour
            calculateCenterOfMass(it)
        }
    }
    private fun findContours(mat: Mat): List<MatOfPoint> {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()
        return contours
    }
    private fun findLargestContour(contours: List<MatOfPoint>): MatOfPoint? {
        return contours.maxByOrNull { Imgproc.contourArea(it) }
    }
    private fun drawContour(mat: Mat, contour: MatOfPoint) {
        Imgproc.drawContours(mat, listOf(contour), -1, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)
    }
    private fun calculateCenterOfMass(contour: MatOfPoint): Point {
        val moments = Imgproc.moments(contour)
        val centerX = moments.m10 / moments.m00
        val centerY = moments.m01 / moments.m00
        return Point(centerX, centerY)
    }
}

object YOLOHelper {
    fun parseTFLiteOutputTensor(
        outputArray: Array<Array<FloatArray>>,
        originalWidth: Int,
        originalHeight: Int
    ): Pair<List<BoundingBox>, List<Point>> {
        val boundingBoxes = mutableListOf<BoundingBox>()
        val listOfPoints = mutableListOf<Point>()

        val numDetections = outputArray[0][0].size // 3549
        Log.d("YOLOTest", "Total detected objects: $numDetections")

        var bestD = 0
        var bestX = 0f
        var bestY = 0f
        var bestW = 0f
        var bestH = 0f
        var bestC = 0f

        for (i in 0 until numDetections) {
            val xCenterNorm = outputArray[0][0][i]  // Normalized x_center (0-1)
            val yCenterNorm = outputArray[0][1][i]  // Normalized y_center (0-1)
            val widthNorm = outputArray[0][2][i]    // Normalized width (0-1)
            val heightNorm = outputArray[0][3][i]   // Normalized height (0-1)
            val confidence = outputArray[0][4][i]   // Confidence score

            if (confidence > bestC) {
                bestD = i
                bestX = xCenterNorm
                bestY = yCenterNorm
                bestW = widthNorm
                bestH = heightNorm
                bestC = confidence
            }
        }

        Log.d("YOLOTest", "BEST DETECTION $bestD: confidence=${"%.8f".format(bestC)}, x_center=$bestX, y_center=$bestY, width=$bestW, height=$bestH")

        // Convert from normalized values (0-1) to absolute pixel values
        val x1 = (bestX - (bestW / 2)) * originalWidth
        val y1 = (bestY - (bestH / 2)) * originalHeight
        val x2 = (bestX + (bestW / 2)) * originalWidth
        val y2 = (bestY + (bestH / 2)) * originalHeight

        Log.d("YOLOTest", "BOUNDING BOX: x1=${"%.8f".format(x1)}, y1=${"%.8f".format(y1)}, x2=${"%.8f".format(x2)}, y2=${"%.8f".format(y2)}")

        // Add bounding box
        boundingBoxes.add(BoundingBox(x1, y1, x2, y2, bestC, 1))
        listOfPoints.add(Point(bestX.toDouble() * originalWidth, bestY.toDouble() * originalHeight))

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
