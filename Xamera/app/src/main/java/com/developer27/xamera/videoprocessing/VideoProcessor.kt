package com.developer27.xamera.videoprocessing

import android.content.Context
import android.graphics.Bitmap
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
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.util.LinkedList

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
        // You can choose a color for the drawn spline; here blue is used.
        var splineLineColor = Scalar(0.0, 0.0, 255.0)
        var lineThickness = 10
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

/**
 * VideoProcessor applies processing to each camera frame.
 * In the processFrameInternalCONTOUR function the frame is converted to a Mat,
 * preprocessed, and contours are found. If a contour is detected, the center is determined,
 * filtered via a Kalman filter, and the list of points is updated.
 *
 * The TraceRenderer.drawSplineCurve(smoothDataList, originalMat) call draws a spline curve
 * connecting the filtered points directly on the original frame Mat.
 *
 * Finally, the Mat (with the drawn lines) is converted back to a Bitmap, which is then returned.
 */
class VideoProcessor(private val context: Context) {
    private var module: Module? = null

    // For line-drawing (visualization)
    private val rawDataList = LinkedList<Point>()
    private val smoothDataList = LinkedList<Point>()

    private var frameCount = 0

    // Storing final data (if needed for later use)
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

    /**
     * Processes a frame by choosing between YOLO inference and contour detection.
     * In this example, we use contour detection.
     */
    fun processFrame(bitmap: Bitmap, callback: (Bitmap?) -> Unit) {
        CoroutineScope(Dispatchers.Default).launch {
            val result = try {
                processFrameInternalCONTOUR(bitmap)
            } catch (e: Exception) {
                logCat("Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) {
                callback(result)
            }
        }
    }

    private fun processFrameInternalCONTOUR(bitmap: Bitmap): Bitmap? {
        val mat = Mat()
        val originalMat = Mat()
        val resizedMat = Mat()

        return try {
            // Convert bitmap to Mat.
            Utils.bitmapToMat(bitmap, originalMat)

            // Preprocess the frame to enhance light blobs.
            val preprocessedMat = Preprocessing.preprocessFrame(originalMat)

            // Find contours in the preprocessed image.
            val contours = ContourDetection.findContours(preprocessedMat)

            // Find the largest contour.
            val largestContour = ContourDetection.findLargestContour(contours)

            if (largestContour != null) {
                // Optionally, you could draw the raw contour:
                // ContourDetection.drawContour(originalMat, largestContour)

                // Calculate the center of mass of the largest contour.
                val center = ContourDetection.calculateCenterOfMass(largestContour)
                rawDataList.add(center)

                // Apply Kalman filter to the center point.
                val (fx, fy) = KalmanHelper.applyKalmanFilter(center)
                smoothDataList.add(Point(fx, fy))

                // Keep the trace lines limited.
                if (rawDataList.size > Settings.Trace.lineLimit) {
                    rawDataList.pollFirst()
                }
                if (smoothDataList.size > Settings.Trace.lineLimit) {
                    smoothDataList.pollFirst()
                }

                // Draw the smoothed trace (spline curve) on the original image.
                TraceRenderer.drawSplineCurve(smoothDataList, originalMat)
            }

            // Convert the modified Mat (with drawn lines) back to Bitmap.
            val outputBitmap = Bitmap.createBitmap(originalMat.cols(), originalMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(originalMat, outputBitmap)
            outputBitmap
        } catch (e: Exception) {
            logCat("Error processing frame: ${e.message}", e)
            null
        } finally {
            // Ensure resources are released.
            mat.release()
            originalMat.release()
            resizedMat.release()
        }
    }

    // (Optional YOLO-based processing can be placed here.)
    private suspend fun processFrameInternalYOLO(bitmap: Bitmap): Bitmap? {
        // Implementation omitted for brevity.
        return null
    }

    private fun makeSquareAndResize(bitmap: Bitmap): Bitmap {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        val height = mat.rows()
        val width = mat.cols()
        val maxDim = maxOf(height, width)
        val top = (maxDim - height) / 2
        val bottom = maxDim - height - top
        val left = (maxDim - width) / 2
        val right = maxDim - width - left
        val paddedMat = Mat()
        Core.copyMakeBorder(mat, paddedMat, top, bottom, left, right, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0))
        val resizedMat = Mat()
        Imgproc.resize(paddedMat, resizedMat, Size(960.0, 960.0), 0.0, 0.0, Imgproc.INTER_AREA)
        val outputBitmap = Bitmap.createBitmap(960, 960, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(resizedMat, outputBitmap)
        mat.release()
        paddedMat.release()
        resizedMat.release()
        return outputBitmap
    }

    fun testYOLOsingleImage(context: Context) {
        // Implementation for single image testing.
    }

    private fun saveInferenceResult(context: Context, mat: Mat) {
        try {
            val outputBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(mat, outputBitmap)
            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val outputFile = File(downloadsDir, "yolo_inference_result.jpg")
            FileOutputStream(outputFile).use { fos ->
                outputBitmap.compress(Bitmap.CompressFormat.JPEG, 90, fos)
                fos.flush()
            }
            Log.d("YOLOTest", "Saved inference result at: ${outputFile.absolutePath}")
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
    // Draws a spline curve connecting the provided points.
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
        Imgproc.threshold(image, thresholdMat, Settings.Brightness.threshold, 255.0, Imgproc.THRESH_TOZERO)
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
        val numDetections = outputTensor.shape()[2].toInt() // For example, 18900
        Log.d("YOLOTest", "Total detected objects: $numDetections")
        var bestD = 0
        var bestX = 0f
        var bestY = 0f
        var bestW = 0f
        var bestH = 0f
        var bestC = 0f
        for (i in 0 until numDetections) {
            val x_center = outputArray[i]
            val y_center = outputArray[i + (numDetections * 1)]
            val width = outputArray[i + (numDetections * 2)]
            val height = outputArray[i + (numDetections * 3)]
            val confidence = outputArray[i + (numDetections * 4)]
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
        val x1 = bestX - (bestW / 2)
        val y1 = bestY - (bestH / 2)
        val x2 = bestX + (bestW / 2)
        val y2 = bestY + (bestH / 2)
        val scaleX = originalWidth.toFloat() / 960f
        val scaleY = originalHeight.toFloat() / 960f
        val scaledX1 = x1 * scaleX
        val scaledY1 = y1 * scaleY
        val scaledX2 = x2 * scaleX
        val scaledY2 = y2 * scaleY
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
            Imgproc.rectangle(mat, Point(textX.toDouble(), textY + baseline[0].toDouble()),
                Point((textX + textSize.width).toDouble(), (textY - textSize.height).toDouble()),
                Settings.BoundingBox.boxColor, Imgproc.FILLED)
            Imgproc.putText(mat, label, Point(textX.toDouble(), textY.toDouble()), Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, Scalar(255.0, 255.0, 255.0), thickness)
        }
    }
}
