package com.developer27.xamera.videoprocessing

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator
import org.opencv.android.OpenCVLoader
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
import java.util.LinkedList
import kotlin.math.roundToInt

/**
 * Example advanced VideoProcessor using OpenCV + Kalman filter.
 * Called in real-time from MainActivity for demonstration/tracking overlay.
 */
data class FrameData(
    val x: Double,
    val y: Double,
    val area: Double,
    val frameCount: Int
)

object Settings {
    object Contour {
        var threshold = 500
    }
    object Trace {
        var lineLimit = 50
        var splineStep = 0.01
        var originalLineColor = Scalar(255.0, 0.0, 0.0) // Red
        var splineLineColor = Scalar(0.0, 0.0, 255.0)  // Blue
        var lineThickness = 4
    }
    object Brightness {
        var factor = 2.0
        var threshold = 150.0
    }
    object Debug {
        var enableToasts = true
        var enableLogging = false
    }
}

class VideoProcessor(private val context: Context) {
    private lateinit var kalmanFilter: KalmanFilter
    private var module: Module? = null

    // For line-drawing (visualization)
    private val rawDataList = LinkedList<Point>()
    private val smoothDataList = LinkedList<Point>()

    // Storing final data
    private val preFilter4Ddata = mutableListOf<FrameData>()
    private val postFilter4Ddata = mutableListOf<FrameData>()

    private var frameCount = 0

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
    }

    fun clearTrackingData() {
        frameCount = 0
        preFilter4Ddata.clear()
        postFilter4Ddata.clear()
        rawDataList.clear()
        smoothDataList.clear()
        showToast("Tracking data reset.")
    }

    // TODO <Soham Naik>: This part will be exported by YOLO to OpenGL
    // ------------------------------------------------------------------------------------
    // **NEW**: Provide final data to MainActivity
    // ------------------------------------------------------------------------------------
    fun getPostFilterData(): List<FrameData> {
        return postFilter4Ddata.toList()
    }

    /**
     * Main processing of each frame
     */
    fun processFrame(bitmap: Bitmap): Bitmap? {
        return try {
            val mat = ImageUtils.bitmapToMat(bitmap)
            val cleanedMat = preprocessFrame(mat)
            mat.release()

            val (centerInfo, processedMat) = detectContourBlob(cleanedMat)
            cleanedMat.release()

            val (center, area) = centerInfo
            if (center != null) {
                val frameData = FrameData(center.x, center.y, area ?: 0.0, frameCount++)
                rawDataList.add(center)
                preFilter4Ddata.add(frameData)

                // Kalman filter
                val (fx, fy) = applyKalmanFilter(center)
                val smoothPoint = Point(fx, fy)
                smoothDataList.add(smoothPoint)
                postFilter4Ddata.add(
                    FrameData(smoothPoint.x, smoothPoint.y, frameData.area, frameData.frameCount)
                )

                // Keep the trace lines limited
                if (rawDataList.size > Settings.Trace.lineLimit) {
                    rawDataList.pollFirst()
                }
                if (smoothDataList.size > Settings.Trace.lineLimit) {
                    smoothDataList.pollFirst()
                }

                // Draw lines
                TraceRenderer.drawRawTrace(rawDataList, processedMat)
                TraceRenderer.drawSplineCurve(smoothDataList, processedMat)
            }

            val outputBitmap = ImageUtils.matToBitmap(processedMat)
            processedMat.release()
            outputBitmap
        } catch (e: Exception) {
            Log.e("VideoProcessor", "Error processing frame: ${e.message}", e)
            null
        }
    }

    private fun applyKalmanFilter(point: Point): Pair<Double, Double> {
        val measurement = Mat(2, 1, CvType.CV_32F).apply {
            put(0, 0, point.x)
            put(1, 0, point.y)
        }
        kalmanFilter.predict()
        val corrected = kalmanFilter.correct(measurement)
        val fx = corrected[0,0][0]
        val fy = corrected[1,0][0]
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

    private fun detectContourBlob(image: Mat): Pair<Pair<Point?, Double?>, Mat> {
        val binaryImage = Mat()
        Imgproc.threshold(image, binaryImage, 200.0, 255.0, Imgproc.THRESH_BINARY)

        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(binaryImage, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val largestContour = contours.maxByOrNull { Imgproc.contourArea(it) }
        val outputImage = Mat()
        Imgproc.cvtColor(image, outputImage, Imgproc.COLOR_GRAY2BGR)

        val areaThreshold = Settings.Contour.threshold.toDouble()
        val (centerPoint, area) = largestContour
            ?.takeIf { Imgproc.contourArea(it) > areaThreshold }
            ?.let {
                Imgproc.drawContours(outputImage, listOf(it), -1, Scalar(255.0, 105.0, 180.0), Imgproc.FILLED)
                val areaVal = Imgproc.contourArea(it)
                val centerVal = calculateCenter(it, outputImage).first
                centerVal to areaVal
            } ?: (null to null)

        binaryImage.release()
        return (centerPoint to area) to outputImage
    }

    private fun calculateCenter(contour: MatOfPoint, image: Mat): Pair<Point?, Pair<Int, Int>?> {
        val moments = Imgproc.moments(contour)
        if (moments.m00 == 0.0) return null to null
        val cx = (moments.m10 / moments.m00).roundToInt()
        val cy = (moments.m01 / moments.m00).roundToInt()
        val centerPoint = Point(cx.toDouble(), cy.toDouble())
        Imgproc.circle(image, centerPoint, 10, Scalar(0.0, 0.0, 255.0), -1)
        return centerPoint to (cx to cy)
    }

    private fun showToast(msg: String) {
        if (Settings.Debug.enableToasts) {
            Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
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
    fun bitmapToMat(bitmap: Bitmap): Mat = Mat().also {
        Utils.bitmapToMat(bitmap, it)
    }
    fun matToBitmap(mat: Mat): Bitmap = Bitmap.createBitmap(
        mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888
    ).apply {
        Utils.matToBitmap(mat, this)
    }
}