@file:Suppress("SameParameterValue")

package com.developer27.xamera.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Environment
import android.util.Log
import android.widget.Toast
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction
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
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.LinkedList
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

// Data class for bounding box information.
data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classId: Int
)

data class ModelDimensions(
    val inputWidth: Int,
    val inputHeight: Int,
    val outputShape: List<Int>
)

// Object to hold various configuration settings.
object Settings {
    object DetectionMode {
        enum class Mode { CONTOUR, YOLO }
        var current: Mode = Mode.YOLO
        var enableYOLOinference = true
    }
    object Trace {
        var enableRAWtrace = true
        var enableSPLINEtrace = true
        var lineLimit = 50
        var splineStep = 0.01
        var originalLineColor = Scalar(255.0, 0.0, 0.0)
        var splineLineColor = Scalar(0.0, 0.0, 255.0)
        var lineThickness = 4
    }
    object BoundingBox {
        var enableBoundingBox = true
        var boxColor = Scalar(0.0, 255.0, 0.0)
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
    object ExportData {
        var frameIMG = false
        var videoDATA = false
    }
}

// Main VideoProcessor class.
class VideoProcessor(private val context: Context) {
    private var tfliteInterpreter: Interpreter? = null
    // List to store raw tracking points.
    private val rawDataList = LinkedList<Point>()
    // List to store smoothed tracking points.
    private val smoothDataList = LinkedList<Point>()
    private var frameCount = 0

    init {
        initOpenCV()
        KalmanHelper.initKalmanFilter()
    }

    // Loads the OpenCV library.
    private fun initOpenCV() {
        try {
            System.loadLibrary("opencv_java4")
            logCat("OpenCV loaded successfully.")
        } catch (e: UnsatisfiedLinkError) {
            logCat("OpenCV failed to load: ${e.message}", e)
        }
    }

    // Sets the TFLite model.
    fun setTFLiteModel(model: Interpreter) {
        synchronized(this) { tfliteInterpreter = model }
        logCat("TFLite Model set in VideoProcessor successfully!")
    }

    // Clears tracking data.
    fun clearTrackingData() {
        frameCount = 0
        rawDataList.clear()
        smoothDataList.clear()
        showToast("Tracking data reset.")
    }

    // Processes a frame asynchronously and returns a Pair (outputBitmap, videoBitmap).
    fun processFrame(bitmap: Bitmap, callback: (Pair<Bitmap, Bitmap>?) -> Unit) {
        CoroutineScope(Dispatchers.Default).launch {
            val result: Pair<Bitmap, Bitmap>? = try {
                when (Settings.DetectionMode.current) {
                    Settings.DetectionMode.Mode.CONTOUR -> {
                        processFrameInternalCONTOUR(bitmap)
                    }
                    Settings.DetectionMode.Mode.YOLO -> {
                        processFrameInternalYOLO(bitmap)
                    }
                }
            } catch (e: Exception) {
                logCat("Error processing frame: ${e.message}", e)
                null
            }
            withContext(Dispatchers.Main) { callback(result) }
        }
    }

    // Processes a frame using Contour Detection - Returns a Pair containing outputBitmap and videoBitmap.
    private fun processFrameInternalCONTOUR(bitmap: Bitmap): Pair<Bitmap, Bitmap>? {
        val originalMat = Mat()
        var preprocessedMat: Mat? = null
        return try {
            Utils.bitmapToMat(bitmap, originalMat)
            preprocessedMat = Preprocessing.preprocessFrame(originalMat)
            val videoBitmap = Bitmap.createBitmap(preprocessedMat.cols(), preprocessedMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(preprocessedMat, videoBitmap)

            val (center, processedMat) = ContourDetection.processContourDetection(preprocessedMat)
            Imgproc.cvtColor(processedMat, processedMat, Imgproc.COLOR_GRAY2BGR)
            if (center != null) {
                updateTrackingData(center, processedMat)
            }
            val outputBitmap = Bitmap.createBitmap(processedMat.cols(), processedMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(processedMat, outputBitmap)

            Pair(outputBitmap, videoBitmap)
        } catch (e: Exception) {
            logCat("Error processing frame: ${e.message}", e)
            null
        } finally {
            originalMat.release()
            preprocessedMat?.release()
        }
    }

    // Processes a frame using YOLO - Returns a Pair containing outputBitmap and letterboxedBitmap.
    private suspend fun processFrameInternalYOLO(bitmap: Bitmap): Pair<Bitmap, Bitmap>? {
        return withContext(Dispatchers.IO) {
            val modelDims = getModelDimensions()
            val origMat = Mat()
            Utils.bitmapToMat(bitmap, origMat)

            Log.e("YOLOTest", "Model Width: ${modelDims.inputWidth}, Model Height: ${modelDims.inputHeight}")

            val (letterboxedMat, padOffsets) = YOLOHelper.letterbox(origMat, modelDims.inputWidth, modelDims.inputHeight)
            origMat.release()

            val preprocessedMat = Preprocessing.preprocessFrame(letterboxedMat)
            letterboxedMat.release()

            val letterboxedBitmap = Bitmap.createBitmap(preprocessedMat.cols(), preprocessedMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(preprocessedMat, letterboxedBitmap)
            preprocessedMat.release()

            val originalMatForDraw = Mat()
            Utils.bitmapToMat(bitmap, originalMatForDraw)

            with(Settings.DetectionMode) {
                if (enableYOLOinference){
                    val tensorImage = TensorImage(DataType.FLOAT32).apply { load(letterboxedBitmap) }
                    if (tfliteInterpreter == null) {
                        Log.e("YOLOTest", "TFLite Model is NULL! Cannot run inference.")
                        return@withContext null
                    }
                    val outputArray = Array(modelDims.outputShape[0]) { Array(modelDims.outputShape[1]) { FloatArray(modelDims.outputShape[2]) } }
                    tfliteInterpreter?.run(tensorImage.buffer, outputArray)
                    val (boundingBox, center) = YOLOHelper.parseTFLiteOutputTensor(outputArray, bitmap.width, bitmap.height, padOffsets, modelDims.inputWidth, modelDims.inputHeight)
                    with(Settings.BoundingBox) {
                        if (enableBoundingBox) YOLOHelper.drawBoundingBoxes(originalMatForDraw, boundingBox)
                    }
                    updateTrackingData(center, originalMatForDraw)
                }
            }

            val outputBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(originalMatForDraw, outputBitmap)
            originalMatForDraw.release()

            Pair(outputBitmap, letterboxedBitmap)
        }
    }

    // Dynamically retrieves the model input size.
    private fun getModelDimensions(): ModelDimensions {
        val inputTensor = tfliteInterpreter?.getInputTensor(0)
        val inputShape = inputTensor?.shape()
        val height = inputShape?.getOrNull(1) ?: 416
        val width = inputShape?.getOrNull(2) ?: 416
        val outputTensor = tfliteInterpreter?.getOutputTensor(0)
        val outputShape: List<Int> = outputTensor?.shape()?.toList() ?: listOf(1, 5, 3549)
        return ModelDimensions(inputWidth = width, inputHeight = height, outputShape = outputShape)
    }

    // Updates tracking data and draws traces on the image.
    private fun updateTrackingData(point: Point, mat: Mat) {
        rawDataList.add(point)
        val (fx, fy) = KalmanHelper.applyKalmanFilter(point)
        smoothDataList.add(Point(fx, fy))
        if (rawDataList.size > Settings.Trace.lineLimit) rawDataList.pollFirst()
        if (smoothDataList.size > Settings.Trace.lineLimit) smoothDataList.pollFirst()
        with(Settings.Trace) {
            if (enableRAWtrace) TraceRenderer.drawRawTrace(smoothDataList, mat)
            if (enableSPLINEtrace) TraceRenderer.drawSplineCurve(smoothDataList, mat)
        }
    }

    // Creates a white, square (28x28) Bitmap that encapsulates the drawn spline trace (with padding).
    fun exportTraceForInference(): Bitmap {
        if (smoothDataList.isEmpty()) {
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.WHITE) }
        }
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE
        var maxY = Double.MIN_VALUE
        for (pt in smoothDataList) {
            minX = min(minX, pt.x)
            minY = min(minY, pt.y)
            maxX = max(maxX, pt.x)
            maxY = max(maxY, pt.y)
        }
        val padding = 30.0
        val optimalWidth = max((maxX - minX + 2 * padding).toInt(), 1)
        val optimalHeight = max((maxY - minY + 2 * padding).toInt(), 1)
        val squareSize = max(optimalWidth, optimalHeight)
        val mat = Mat(squareSize, squareSize, CvType.CV_8UC4, Scalar(255.0, 255.0, 255.0, 255.0))
        val xOffset = (squareSize - optimalWidth) / 2.0
        val yOffset = (squareSize - optimalHeight) / 2.0
        val adjustedPoints = smoothDataList.map {
            Point(it.x - minX + padding + xOffset, it.y - minY + padding + yOffset)
        }
        val originalColor = Settings.Trace.splineLineColor
        val originalThickness = Settings.Trace.lineThickness
        Settings.Trace.splineLineColor = Scalar(0.0, 0.0, 0.0) // Black
        Settings.Trace.lineThickness = 40
        TraceRenderer.drawSplineCurve(adjustedPoints, mat)
        Settings.Trace.splineLineColor = originalColor
        Settings.Trace.lineThickness = originalThickness
        val outputBitmap = Bitmap.createBitmap(squareSize, squareSize, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, outputBitmap)
        mat.release()
        val scaledBitmap = Bitmap.createScaledBitmap(outputBitmap, 28, 28, true)
        return scaledBitmap
    }

    // Shows a Toast message.
    private fun showToast(msg: String) {
        if (Settings.Debug.enableToasts) {
            Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
        }
    }

    // Logs messages to Logcat.
    private fun logCat(message: String, throwable: Throwable? = null) {
        if (Settings.Debug.enableLogging) {
            if (throwable != null) Log.e("VideoProcessor", message, throwable)
            else Log.d("VideoProcessor", message)
        }
    }

    /**
     * Automatically saves the current (smoothed) tracking data—the points that form the drawn line—into a text file.
     * The file name incorporates the current date and time, and the file is saved in the public Documents/tracking folder.
     * The file is saved with a .xmr extension.
     *
     * IMPORTANT: To write to the public Documents folder, you may need to declare and request the
     * WRITE_EXTERNAL_STORAGE permission in your AndroidManifest.xml and at runtime.
     */
    fun autoSaveLineData() {
        try {
            val documentsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            val trackingDir = File(documentsDir, "tracking")
            if (!trackingDir.exists()) {
                trackingDir.mkdirs()
            }
            val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
            val currentDate = dateFormat.format(Date())
            val fileName = "TrackingData_$currentDate.txt"
            val file = File(trackingDir, fileName)
            val dataString = smoothDataList.joinToString(separator = "\n") { point ->
                "${point.x},${point.y}"
            }
            file.writeText(dataString)
            logCat("Tracking data saved to ${file.absolutePath}")
            Toast.makeText(context, "Tracking data saved to ${file.absolutePath}", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            logCat("Error saving tracking data: ${e.message}", e)
            Toast.makeText(context, "Error saving tracking data", Toast.LENGTH_SHORT).show()
        }
    }
}

// Helper object to draw raw and spline traces.
object TraceRenderer {
    fun drawRawTrace(data: List<Point>, image: Mat) {
        for (i in 1 until data.size) {
            Imgproc.line(image, data[i - 1], data[i], Settings.Trace.originalLineColor, Settings.Trace.lineThickness)
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
            prevPoint?.let { Imgproc.line(image, it, currentPoint, Settings.Trace.splineLineColor, Settings.Trace.lineThickness) }
            prevPoint = currentPoint
            t += Settings.Trace.splineStep
        }
    }
}

// Helper object for spline interpolation using Apache Commons Math.
object SplineHelper {
    fun applySplineInterpolation(data: List<Point>): Pair<PolynomialSplineFunction, PolynomialSplineFunction>? {
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

// Helper object for applying a Kalman filter to smooth tracking points.
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

// Helper object for preprocessing frames with OpenCV.
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
    private fun enhanceBrightness(image: Mat): Mat = Mat().apply { Core.multiply(image, Scalar(Settings.Brightness.factor), this) }
    private fun conditionalThresholding(image: Mat): Mat {
        val thresholdMat = Mat()
        Imgproc.threshold(image, thresholdMat, Settings.Brightness.threshold, 255.0, Imgproc.THRESH_TOZERO)
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

// Helper object for contour detection.
object ContourDetection {
    fun processContourDetection(mat: Mat): Pair<Point?, Mat> {
        val contours = findContours(mat)
        val largestContour = findLargestContour(contours)
        val center = largestContour?.let {
            drawContour(mat, it)
            calculateCenterOfMass(it)
        }
        return Pair(center, mat)
    }
    private fun findContours(mat: Mat): List<MatOfPoint> {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()
        return contours
    }
    private fun findLargestContour(contours: List<MatOfPoint>): MatOfPoint? = contours.maxByOrNull { Imgproc.contourArea(it) }
    private fun drawContour(mat: Mat, contour: MatOfPoint) {
        Imgproc.drawContours(mat, listOf(contour), -1, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)
    }
    private fun calculateCenterOfMass(contour: Mat): Point {
        val moments = Imgproc.moments(contour)
        val centerX = moments.m10 / moments.m00
        val centerY = moments.m01 / moments.m00
        return Point(centerX, centerY)
    }
}

// Helper object for YOLO detection using TensorFlow Lite.
object YOLOHelper {
    fun parseTFLiteOutputTensor(outputArray: Array<Array<FloatArray>>, originalWidth: Int, originalHeight: Int, padOffsets: Pair<Int, Int>, modelInputWidth: Int, modelInputHeight: Int): Pair<BoundingBox, Point> {
        val numDetections = outputArray[0][0].size
        Log.d("YOLOTest", "Total detected objects: $numDetections")

        var bestD = 0
        var bestX = 0f
        var bestY = 0f
        var bestW = 0f
        var bestH = 0f
        var bestC = 0f

        for (i in 0 until numDetections) {
            val xCenterNorm = outputArray[0][0][i]
            val yCenterNorm = outputArray[0][1][i]
            val widthNorm = outputArray[0][2][i]
            val heightNorm = outputArray[0][3][i]
            val confidence = outputArray[0][4][i]
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

        val scale = min(modelInputWidth / originalWidth.toDouble(), modelInputHeight / originalHeight.toDouble())
        val padLeft = padOffsets.first.toDouble()
        val padTop = padOffsets.second.toDouble()
        val xCenterLetterboxed = bestX * modelInputWidth
        val yCenterLetterboxed = bestY * modelInputHeight
        val boxWidthLetterboxed = bestW * modelInputWidth
        val boxHeightLetterboxed = bestH * modelInputHeight
        val xCenterOriginal = (xCenterLetterboxed - padLeft) / scale
        val yCenterOriginal = (yCenterLetterboxed - padTop) / scale
        val boxWidthOriginal = boxWidthLetterboxed / scale
        val boxHeightOriginal = boxHeightLetterboxed / scale
        val x1Original = xCenterOriginal - (boxWidthOriginal / 2)
        val y1Original = yCenterOriginal - (boxHeightOriginal / 2)
        val x2Original = xCenterOriginal + (boxWidthOriginal / 2)
        val y2Original = yCenterOriginal + (boxHeightOriginal / 2)
        Log.d("YOLOTest", "Adjusted BOUNDING BOX: x1=${"%.8f".format(x1Original)}, y1=${"%.8f".format(y1Original)}, x2=${"%.8f".format(x2Original)}, y2=${"%.8f".format(y2Original)}")
        val boundingBox = BoundingBox(x1Original.toFloat(), y1Original.toFloat(), x2Original.toFloat(), y2Original.toFloat(), bestC, 1)
        val center = Point(xCenterOriginal, yCenterOriginal)
        return Pair(boundingBox, center)
    }
    fun drawBoundingBoxes(mat: Mat, box: BoundingBox) {
        val topLeft = Point(box.x1.toDouble(), box.y1.toDouble())
        val bottomRight = Point(box.x2.toDouble(), box.y2.toDouble())
        Imgproc.rectangle(mat, topLeft, bottomRight, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)
        val label = "User_1 (${"%.2f".format(box.confidence * 100)}%)"
        val fontScale = 0.6
        val thickness = 1
        val baseline = IntArray(1)
        val textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, thickness, baseline)
        val textX = box.x1.toInt()
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
    fun letterbox(src: Mat, targetWidth: Int, targetHeight: Int, padColor: Scalar = Scalar(0.0, 0.0, 0.0)): Pair<Mat, Pair<Int, Int>> {
        val srcWidth = src.cols().toDouble()
        val srcHeight = src.rows().toDouble()
        val scale = min(targetWidth / srcWidth, targetHeight / srcHeight)
        val newWidth = (srcWidth * scale).toInt()
        val newHeight = (srcHeight * scale).toInt()
        val resized = Mat()
        Imgproc.resize(src, resized, Size(newWidth.toDouble(), newHeight.toDouble()))
        val padWidth = targetWidth - newWidth
        val padHeight = targetHeight - newHeight
        val top = padHeight / 2
        val bottom = padHeight - top
        val left = padWidth / 2
        val right = padWidth - left
        val letterboxed = Mat()
        Core.copyMakeBorder(resized, letterboxed, top, bottom, left, right, Core.BORDER_CONSTANT, padColor)
        return Pair(letterboxed, Pair(left, top))
    }
}
