@file:Suppress("SameParameterValue")

package com.developer27.xamera.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
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
import java.util.LinkedList
import kotlin.math.max
import kotlin.math.min

data class DetectionResult(
    val xCenter: Float,
    val yCenter: Float,
    val width: Float,
    val height: Float,
    val confidence: Float
)

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
    object Inference {
        var confidenceThreshold: Float = 0.5f
        var iouThreshold: Float = 0.5f
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
    private var tfliteInterpreter_YOLO: Interpreter? = null
    private var tfliteInterpreter_DIGIT: Interpreter? = null
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
    fun setYOLOmodel(model: Interpreter) {
        synchronized(this) { tfliteInterpreter_YOLO = model }
        logCat("TFLite Model set in VideoProcessor successfully!")
    }

    fun setDigitModel(model: Interpreter) {
        synchronized(this) { tfliteInterpreter_DIGIT = model }
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
                    Settings.DetectionMode.Mode.CONTOUR -> processFrameInternalCONTOUR(bitmap)
                    Settings.DetectionMode.Mode.YOLO -> processFrameInternalYOLO(bitmap)
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
                rawDataList.add(center)
                val (fx, fy) = KalmanHelper.applyKalmanFilter(center)
                smoothDataList.add(Point(fx, fy))
                if (rawDataList.size > Settings.Trace.lineLimit) rawDataList.pollFirst()
                if (smoothDataList.size > Settings.Trace.lineLimit) smoothDataList.pollFirst()
                with(Settings.Trace) {
                    if (enableRAWtrace) TraceRenderer.drawRawTrace(smoothDataList, processedMat)
                    if (enableSPLINEtrace) TraceRenderer.drawSplineCurve(smoothDataList, processedMat)
                }
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
            // Retrieve model dimensions (input size and output shape) in one go.
            val modelDims = getModelDimensions()  // e.g., inputWidth = 416, inputHeight = 416, outputShape = [1, 5, 3549]

            // Convert original bitmap to a Mat.
            val origMat = Mat()
            Utils.bitmapToMat(bitmap, origMat)

            Log.e("YOLOTest", "Model Width: ${modelDims.inputWidth}, Model Height: ${modelDims.inputHeight}")

            // Apply letterbox: resize and pad to the target dimensions.
            val (letterboxedMat, padOffsets) = YOLOHelper.letterbox(origMat, modelDims.inputWidth, modelDims.inputHeight)
            origMat.release()

            // Preprocess the letterboxed image.
            val preprocessedMat = Preprocessing.preprocessFrame(letterboxedMat)
            letterboxedMat.release()

            // Convert the preprocessed Mat back to a Bitmap.
            val letterboxedBitmap = Bitmap.createBitmap(preprocessedMat.cols(), preprocessedMat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(preprocessedMat, letterboxedBitmap)
            preprocessedMat.release()

            // Create a Mat from the original bitmap for drawing bounding boxes.
            val originalMatForDraw = Mat()
            Utils.bitmapToMat(bitmap, originalMatForDraw)

            with(Settings.DetectionMode) {
                if (enableYOLOinference) {
                    // Prepare tensor image for inference.
                    val tensorImage = TensorImage(DataType.FLOAT32).apply { load(letterboxedBitmap) }
                    if (tfliteInterpreter_YOLO == null) {
                        Log.e("YOLOTest", "TFLite Model is NULL! Cannot run inference.")
                        return@withContext null
                    }
                    // Allocate output array using the output shape from modelDims.
                    val outputArray = Array(modelDims.outputShape[0]) { Array(modelDims.outputShape[1]) { FloatArray(modelDims.outputShape[2]) } }
                    // Run inference.
                    tfliteInterpreter_YOLO?.run(tensorImage.buffer, outputArray)
                    // Finds Best Detection
                    val detectionResult = YOLOHelper.parseTFLite(outputArray)
                    // Based on Detection, scale to Output Screen and output Center and Bounding Box
                    detectionResult?.let { detection ->
                        // Rescale detection to original image space.
                        val (boundingBox, center) = YOLOHelper.rescaleInferencedCoordinates(
                            detection, bitmap.width, bitmap.height, padOffsets,
                            modelDims.inputWidth, modelDims.inputHeight
                        )
                        if (Settings.BoundingBox.enableBoundingBox) {
                            YOLOHelper.drawBoundingBoxes(originalMatForDraw, boundingBox)
                        }
                        // Update tracking data.
                        rawDataList.add(center)
                        val (fx, fy) = KalmanHelper.applyKalmanFilter(center)
                        smoothDataList.add(Point(fx, fy))
                        // Keep the list sizes within the specified limits.
                        while (rawDataList.size > Settings.Trace.lineLimit) rawDataList.pollFirst()
                        while (smoothDataList.size > Settings.Trace.lineLimit) smoothDataList.pollFirst()
                    }

                    // Always render the trace lines even if no new detection was added.
                    with(Settings.Trace) {
                        if (enableRAWtrace) TraceRenderer.drawRawTrace(smoothDataList, originalMatForDraw)
                        if (enableSPLINEtrace) TraceRenderer.drawSplineCurve(smoothDataList, originalMatForDraw)
                    }
                }
            }
            // Convert the annotated Mat back to a Bitmap.
            val outputBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(originalMatForDraw, outputBitmap)
            originalMatForDraw.release()
            // - First: the outputBitmap (original image with drawn bounding boxes)
            // - Second: the letterboxedBitmap (the preprocessed input used for inference)
            Pair(outputBitmap, letterboxedBitmap)
        }
    }

    // Dynamically retrieves the model input size.
    private fun getModelDimensions(): ModelDimensions {
        // Retrieve input tensor shape.
        val inputTensor = tfliteInterpreter_YOLO?.getInputTensor(0)
        val inputShape = inputTensor?.shape()
        // Typically, the input tensor shape is [1, height, width, channels].
        val height = inputShape?.getOrNull(1) ?: 416
        val width = inputShape?.getOrNull(2) ?: 416

        // Retrieve output tensor shape.
        val outputTensor = tfliteInterpreter_YOLO?.getOutputTensor(0)
        val outputShape: List<Int> = outputTensor?.shape()?.toList() ?: listOf(1, 5, 3549)

        return ModelDimensions(inputWidth = width, inputHeight = height, outputShape = outputShape)
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

    // Creates a white, square (28x28) Bitmap that encapsulates the drawn spline trace (with padding).
    fun exportTraceForInference(): Bitmap {
        // Ensure there is some trace data.
        if (smoothDataList.isEmpty()) {
            // Return a minimal white bitmap if there's nothing to draw.
            return Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.WHITE) }
        }

        // 1. Compute the bounding box of the trace points.
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

        // 2. Define padding (in pixels) around the drawn trace.
        val padding = 30.0
        // Compute optimal dimensions.
        val optimalWidth = max((maxX - minX + 2 * padding).toInt(), 1)
        val optimalHeight = max((maxY - minY + 2 * padding).toInt(), 1)

        // 3. Determine the square size as the greatest of the optimal dimensions.
        val squareSize = max(optimalWidth, optimalHeight)

        // 4. Create a white square Mat of the computed dimensions.
        val mat = Mat(squareSize, squareSize, CvType.CV_8UC4, Scalar(255.0, 255.0, 255.0, 255.0))

        // 5. Compute offsets to center the drawn trace inside the square.
        val xOffset = (squareSize - optimalWidth) / 2.0
        val yOffset = (squareSize - optimalHeight) / 2.0

        // 6. Create an adjusted list of points so that the drawing starts at (padding, padding) plus the offsets.
        val adjustedPoints = smoothDataList.map {
            Point(it.x - minX + padding + xOffset, it.y - minY + padding + yOffset)
        }

        // 7. Set up drawing parameters (temporarily override settings).
        val originalColor = Settings.Trace.splineLineColor
        val originalThickness = Settings.Trace.lineThickness
        Settings.Trace.splineLineColor = Scalar(0.0, 0.0, 0.0) // Black
        Settings.Trace.lineThickness = 40

        // 8. Draw the spline curve using the adjusted points.
        TraceRenderer.drawSplineCurve(adjustedPoints, mat)

        // 9. Restore the original settings.
        Settings.Trace.splineLineColor = originalColor
        Settings.Trace.lineThickness = originalThickness

        // 10. Convert the Mat back to a Bitmap.
        val outputBitmap = Bitmap.createBitmap(squareSize, squareSize, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, outputBitmap)
        mat.release()
        val scaledBitmap = Bitmap.createScaledBitmap(outputBitmap, 28, 28, true)
        return scaledBitmap
    }

    /**
     * Returns the tracking coordinates as a semicolon-separated string.
     * Each point is formatted as "x,y,0.0".
     */
    fun getTrackingCoordinatesString(): String {
        return smoothDataList.joinToString(separator = ";") { "${it.x},${it.y},0.0" }
    }

    fun getDigitInterpreter(): Interpreter? {
        return tfliteInterpreter_DIGIT
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
    // Finds the best detection (highest confidence) from the TFLite output.
    fun parseTFLite(rawOutput: Array<Array<FloatArray>>): DetectionResult? {
        val numDetections = rawOutput[0][0].size
        Log.d("YOLOTest", "Total detected objects: $numDetections")

        // Step 1: Parse detections and filter by confidence.
        val detections = mutableListOf<DetectionResult>()
        for (i in 0 until numDetections) {
            val xCenter = rawOutput[0][0][i]
            val yCenter = rawOutput[0][1][i]
            val width = rawOutput[0][2][i]
            val height = rawOutput[0][3][i]
            val confidence = rawOutput[0][4][i]
            if (confidence >= Settings.Inference.confidenceThreshold) {
                detections.add(DetectionResult(xCenter, yCenter, width, height, confidence))
            }
        }
        if (detections.isEmpty()) {
            Log.d("YOLOTest", "No detections above confidence threshold: ${Settings.Inference.confidenceThreshold}")
            return null
        }

        // Step 2: Convert detections to bounding boxes.
        val detectionBoxes = detections.map { it to detectionToBox(it) }.toMutableList()
        // Sort by confidence (highest first).
        detectionBoxes.sortByDescending { it.first.confidence }

        // Step 3: Apply NMS.
        val nmsDetections = mutableListOf<DetectionResult>()
        while (detectionBoxes.isNotEmpty()) {
            val current = detectionBoxes.removeAt(0)
            nmsDetections.add(current.first)
            detectionBoxes.removeAll { other ->
                computeIoU(current.second, other.second) > Settings.Inference.iouThreshold
            }
        }

        // Step 4: Choose the detection with the highest confidence from the remaining.
        val bestDetection = nmsDetections.maxByOrNull { it.confidence }
        bestDetection?.let { d ->
            Log.d(
                "YOLOTest",
                "BEST DETECTION: confidence=${"%.8f".format(d.confidence)}, x_center=${d.xCenter}, y_center=${d.yCenter}, width=${d.width}, height=${d.height}"
            )
        }
        return bestDetection
    }

    // Helper: Convert a detection to a bounding box in normalized coordinates.
    private fun detectionToBox(d: DetectionResult): BoundingBox {
        val x1 = d.xCenter - d.width / 2
        val y1 = d.yCenter - d.height / 2
        val x2 = d.xCenter + d.width / 2
        val y2 = d.yCenter + d.height / 2
        return BoundingBox(x1, y1, x2, y2, d.confidence, 1)
    }

    // Helper: Compute IoU between two normalized bounding boxes.
    private fun computeIoU(boxA: BoundingBox, boxB: BoundingBox): Float {
        val x1 = max(boxA.x1, boxB.x1)
        val y1 = max(boxA.y1, boxB.y1)
        val x2 = min(boxA.x2, boxB.x2)
        val y2 = min(boxA.y2, boxB.y2)

        val intersectionWidth = max(0f, x2 - x1)
        val intersectionHeight = max(0f, y2 - y1)
        val intersectionArea = intersectionWidth * intersectionHeight

        val areaA = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1)
        val areaB = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1)
        val unionArea = areaA + areaB - intersectionArea

        return if (unionArea > 0f) intersectionArea / unionArea else 0f
    }

    // Takes the raw detection and rescales its coordinates to the original image.
    fun rescaleInferencedCoordinates(
        detection: DetectionResult,
        originalWidth: Int,
        originalHeight: Int,
        padOffsets: Pair<Int, Int>,
        modelInputWidth: Int,
        modelInputHeight: Int
    ): Pair<BoundingBox, Point> {
        // Compute the scale factor used in the letterbox transformation.
        val scale = min(modelInputWidth / originalWidth.toDouble(), modelInputHeight / originalHeight.toDouble())

        // Get the padding applied during letterboxing.
        val padLeft = padOffsets.first.toDouble()
        val padTop = padOffsets.second.toDouble()

        // Convert normalized coordinates to letterboxed image coordinates.
        val xCenterLetterboxed = detection.xCenter * modelInputWidth
        val yCenterLetterboxed = detection.yCenter * modelInputHeight
        val boxWidthLetterboxed = detection.width * modelInputWidth
        val boxHeightLetterboxed = detection.height * modelInputHeight

        // Remove padding and rescale back to original image coordinates.
        val xCenterOriginal = (xCenterLetterboxed - padLeft) / scale
        val yCenterOriginal = (yCenterLetterboxed - padTop) / scale
        val boxWidthOriginal = boxWidthLetterboxed / scale
        val boxHeightOriginal = boxHeightLetterboxed / scale

        // Compute bounding box corners in original image coordinates.
        val x1Original = xCenterOriginal - (boxWidthOriginal / 2)
        val y1Original = yCenterOriginal - (boxHeightOriginal / 2)
        val x2Original = xCenterOriginal + (boxWidthOriginal / 2)
        val y2Original = yCenterOriginal + (boxHeightOriginal / 2)

        Log.d("YOLOTest", "Adjusted BOUNDING BOX: x1=${"%.8f".format(x1Original)}, y1=${"%.8f".format(y1Original)}, x2=${"%.8f".format(x2Original)}, y2=${"%.8f".format(y2Original)}")

        // Create the bounding box and center point objects.
        val boundingBox = BoundingBox(
            x1Original.toFloat(),
            y1Original.toFloat(),
            x2Original.toFloat(),
            y2Original.toFloat(),
            detection.confidence,
            1 // Class index (or whatever label you're using)
        )
        val center = Point(xCenterOriginal, yCenterOriginal)
        return Pair(boundingBox, center)
    }

    fun drawBoundingBoxes(mat: Mat, box: BoundingBox) {
        val topLeft = Point(box.x1.toDouble(), box.y1.toDouble())
        val bottomRight = Point(box.x2.toDouble(), box.y2.toDouble())
        Imgproc.rectangle(mat, topLeft, bottomRight, Settings.BoundingBox.boxColor, Settings.BoundingBox.boxThickness)
        val label = "User_1 (${("%.2f".format(box.confidence * 100))}%)"
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
    // Resizes an image while maintaining its aspect ratio and pads it to fit the target dimensions.
    fun letterbox(src: Mat, targetWidth: Int, targetHeight: Int, padColor: Scalar = Scalar(0.0, 0.0, 0.0)): Pair<Mat, Pair<Int, Int>> {
        val srcWidth = src.cols().toDouble()
        val srcHeight = src.rows().toDouble()
        // Compute scaling factor: use the smaller ratio
        val scale = min(targetWidth / srcWidth, targetHeight / srcHeight)
        val newWidth = (srcWidth * scale).toInt()
        val newHeight = (srcHeight * scale).toInt()

        // Resize the source image
        val resized = Mat()
        Imgproc.resize(src, resized, Size(newWidth.toDouble(), newHeight.toDouble()))

        // Compute padding needed to reach target dimensions
        val padWidth = targetWidth - newWidth
        val padHeight = targetHeight - newHeight
        val top = padHeight / 2
        val bottom = padHeight - top
        val left = padWidth / 2
        val right = padWidth - left

        // Create the final letterboxed image with padding
        val letterboxed = Mat()
        Core.copyMakeBorder(resized, letterboxed, top, bottom, left, right, Core.BORDER_CONSTANT, padColor)

        // Return the letterboxed image and the top-left padding offset.
        return Pair(letterboxed, Pair(left, top))
    }
}