// File: VideoProcessor.kt
// Written by Soham Naik
// Last Updated 12/13/2024
package com.developer27.xamera

import org.opencv.android.Utils
import android.graphics.Bitmap
import android.content.Context
import android.util.Log
import android.widget.Toast
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.video.KalmanFilter
import java.util.*
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction

/**
 * Global application settings.
 */
object Settings {
    object Contour {
        /** The minimum area threshold for contour detection. */
        var threshold = 500
    }
    object Trace {
        /** The maximum number of raw points to store and display. (No limit: lines remain indefinitely) */
        var lineLimit = Int.MAX_VALUE
        /** Step size for spline evaluation. Smaller values yield smoother curves. */
        var splineStep = 0.01
        /** Color of the raw trace line (red). */
        var originalLineColor = Scalar(255.0, 0.0, 0.0)
        /** Color of the spline curve line (blue). */
        var splineLineColor = Scalar(0.0, 0.0, 255.0)
        /** Thickness of the trace and spline lines. */
        var lineThickness = 4
    }
    object Brightness {
        /** Factor by which the brightness is enhanced. */
        var factor = 2.0
        /** Threshold to apply for conditional thresholding. */
        var threshold = 150.0
    }
    object Debug {
        /** Enable user-facing toasts for debugging. */
        var enableToasts = true
        /** Enable log debugging statements. */
        var enableLogging = true
    }
}

/**
 * Data class representing frame-level information about a detected point.
 *
 * @param x The x-coordinate of the detected point.
 * @param y The y-coordinate of the detected point.
 * @param area The area of the detected contour at this frame.
 * @param frameCount The sequential frame number.
 */
data class FrameData(val x: Double, val y: Double, val area: Double, val frameCount: Int)

/**
 * The VideoProcessor class handles frame-by-frame image analysis, point tracking,
 * data filtering (via Kalman filter), and delegates rendering (raw trace and spline) to TraceRenderer.
 *
 * @param context The Android context used for UI feedback (toasts, logs).
 */
class VideoProcessor(private val context: Context) {

    private lateinit var kalmanFilter: KalmanFilter

    /**
     * The list of raw detected points. No removal is done so lines remain on the screen until "Stop Tracking".
     */
    private val centerDataList = LinkedList<Point>()

    /**
     * Lists for storing frame data before and after Kalman filtering.
     */
    private val preFilter4Ddata = mutableListOf<FrameData>()
    private val postFilter4Ddata = mutableListOf<FrameData>()

    private var frameCount = 0

    init {
        initializeOpenCV()
    }

    /**
     * Attempts to load and initialize OpenCV and the Kalman filter.
     */
    private fun initializeOpenCV() {
        if (OpenCVLoader.initDebug()) {
            showToast("OpenCV loaded successfully")
            initializeKalmanFilter()
        } else {
            // Optionally handle initialization failure
        }
    }

    /**
     * Initializes the Kalman filter parameters for smoothing detections.
     */
    private fun initializeKalmanFilter() {
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

    /**
     * Clears all tracking data and resets the frame count.
     * This also removes all lines from the screen once "Stop Tracking" is pressed by the user.
     */
    fun clearTrackingData() {
        frameCount = 0
        preFilter4Ddata.clear()
        postFilter4Ddata.clear()
        centerDataList.clear()
        showToast("Tracking started: data reset.")
    }

    /**
     * Processes a single frame asynchronously. Performs preprocessing, contour detection,
     * Kalman filtering, and updates the trace.
     *
     * @param bitmap The input frame as a Bitmap.
     * @return The processed frame as a Bitmap with overlays, or null if processing fails.
     */
    suspend fun processFrame(bitmap: Bitmap): Bitmap? = withContext(Dispatchers.Default) {
        try {
            val mat = ImageUtils.bitmapToMat(bitmap)
            // Apply preprocessing steps
            val grayMat = Preprocessing.applyGrayscale(mat).also { mat.release() }
            val enhancedMat = Preprocessing.enhanceBrightness(grayMat).also { grayMat.release() }
            val thresholdMat = Preprocessing.conditionalThresholding(enhancedMat).also { enhancedMat.release() }
            val blurredMat = Preprocessing.applyGaussianBlur(thresholdMat).also { thresholdMat.release() }
            val cleanedMat = Preprocessing.applyMorphologicalClosing(blurredMat).also { blurredMat.release() }

            // Detect the largest contour blob and compute its center
            val (centerInfo, processedMat) = detectContourBlob(cleanedMat).also { cleanedMat.release() }
            val (center, area) = centerInfo

            // If a center is found, process and update the trace
            center?.let {
                val frameData = FrameData(it.x, it.y, area ?: 0.0, frameCount++)
                preFilter4Ddata.add(frameData)

                // Apply Kalman filter to get filtered coordinates
                val (filteredX, filteredY) = applyKalmanFilter(it, area, frameData.frameCount)
                val filteredPoint = Point(filteredX, filteredY)

                // Add the filtered point without removing old ones
                centerDataList.add(filteredPoint)

                // Delegate drawing to TraceRenderer
                TraceRenderer.drawRawTrace(centerDataList, processedMat)
                TraceRenderer.drawSplineCurve(centerDataList, processedMat)
            }

            return@withContext ImageUtils.matToBitmap(processedMat).also { processedMat.release() }
        } catch (e: Exception) {
            Log.e("VideoProcessor", "Error processing frame: ${e.message}")
            e.printStackTrace()
            return@withContext null
        }
    }

    /**
     * Applies the Kalman filter to reduce jitter and returns the filtered coordinates.
     */
    private fun applyKalmanFilter(center: Point, area: Double?, frame: Int): Pair<Double, Double> {
        val measurement = Mat(2,1,CvType.CV_32F).apply {
            put(0,0, center.x)
            put(1,0, center.y)
        }

        kalmanFilter.predict()
        val corrected = kalmanFilter.correct(measurement)
        val filteredX = corrected[0,0][0]
        val filteredY = corrected[1,0][0]
        val filteredFrameData = FrameData(filteredX, filteredY, area ?: 0.0, frame)
        postFilter4Ddata.add(filteredFrameData)
        return Pair(filteredX, filteredY)
    }

    /**
     * Detects the largest contour in the given image. Returns the center of that contour (if any) and the processed image.
     */
    private fun detectContourBlob(image: Mat): Pair<Pair<Point?, Double?>, Mat> {
        val binaryImage = Mat()
        Imgproc.threshold(image, binaryImage, 200.0, 255.0, Imgproc.THRESH_BINARY)

        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(binaryImage, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val largestContour = contours.maxByOrNull { Imgproc.contourArea(it) }
        val outputImage = Mat()
        Imgproc.cvtColor(image, outputImage, Imgproc.COLOR_GRAY2BGR)

        val (centerPoint, area) = largestContour?.takeIf { Imgproc.contourArea(it) > Settings.Contour.threshold }
            ?.let {
                Imgproc.drawContours(outputImage, listOf(it), -1, Scalar(255.0, 105.0, 180.0), Imgproc.FILLED)
                val area = Imgproc.contourArea(it)
                val center = calculateCenter(it, outputImage).first
                Pair(center, area)
            } ?: Pair(null, null)

        binaryImage.release()
        return Pair(Pair(centerPoint, area), outputImage)
    }

    /**
     * Calculates the center of a given contour using image moments.
     */
    private fun calculateCenter(contour: MatOfPoint, image: Mat): Pair<Point?, Pair<Int, Int>?> {
        val moments = Imgproc.moments(contour)
        return if (moments.m00 != 0.0) {
            val centerX = (moments.m10 / moments.m00).toInt()
            val centerY = (moments.m01 / moments.m00).toInt()
            val centerPoint = Point(centerX.toDouble(), centerY.toDouble())
            // Draw center in red: B=0,G=0,R=255
            Imgproc.circle(image, centerPoint, 10, Scalar(0.0,0.0,255.0), -1)
            Pair(centerPoint, Pair(centerX, centerY))
        } else Pair(null, null)
    }

    /**
     * Returns the raw frame data before filtering.
     */
    fun retrievePreFilter4Ddata(): List<FrameData> = preFilter4Ddata

    /**
     * Returns the filtered frame data after applying the Kalman filter.
     */
    fun retrievePostFilter4Ddata(): List<FrameData> = postFilter4Ddata

    /**
     * Displays a toast message if debugging is enabled.
     */
    private fun showToast(message: String) {
        if (Settings.Debug.enableToasts) {
            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Logs a debug message if debugging is enabled.
     */
    private fun logDebug(message: String) {
        if (Settings.Debug.enableLogging) {
            Log.d("VideoProcessor", message)
        }
    }
}

/**
 * Object responsible for rendering operations such as drawing the raw trace and the spline curve.
 */
object TraceRenderer {
    /**
     * Draws the original (raw) detected points trace.
     *
     * @param data The list of points representing the raw trace.
     * @param image The image on which to draw.
     */
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

    /**
     * Draws a spline curve based on the data points.
     * Uses SplineHelper to generate spline functions.
     *
     * @param data The list of points from which to compute the spline.
     * @param image The image on which to draw the spline curve.
     */
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

/**
 * Helper object for spline interpolation logic.
 */
object SplineHelper {
    /**
     * Applies spline interpolation to generate spline functions for the given data points.
     *
     * @param data The list of points for which spline interpolation is computed.
     * @return A pair of PolynomialSplineFunctions (splineX, splineY) if successful, otherwise null.
     */
    fun applySplineInterpolation(data: List<Point>): Pair<PolynomialSplineFunction, PolynomialSplineFunction>? {
        if (data.size < 2) return null
        val interpolator = SplineInterpolator()
        val xData = data.map { it.x }.toDoubleArray()
        val yData = data.map { it.y }.toDoubleArray()
        val tData = data.indices.map { it.toDouble() }.toDoubleArray()
        val splineX = interpolator.interpolate(tData, xData)
        val splineY = interpolator.interpolate(tData, yData)
        return Pair(splineX, splineY)
    }
}

/**
 * A utility class that provides preprocessing functions for image frames.
 */
class Preprocessing {
    companion object {
        /**
         * Converts a BGR image to grayscale.
         */
        fun applyGrayscale(frame: Mat): Mat {
            val grayMat = Mat()
            Imgproc.cvtColor(frame, grayMat, Imgproc.COLOR_BGR2GRAY)
            return grayMat
        }

        /**
         * Enhances the brightness of an image by multiplying pixel values.
         */
        fun enhanceBrightness(image: Mat): Mat = Mat().apply {
            Core.multiply(image, Scalar(Settings.Brightness.factor), this)
        }

        /**
         * Applies a conditional thresholding to highlight regions above a certain brightness threshold.
         */
        fun conditionalThresholding(image: Mat): Mat {
            val thresholdMat = Mat()
            Imgproc.threshold(
                image,
                thresholdMat,
                Settings.Brightness.threshold,
                255.0,
                Imgproc.THRESH_TOZERO
            )
            return thresholdMat
        }

        /**
         * Applies a Gaussian blur to reduce noise.
         */
        fun applyGaussianBlur(image: Mat): Mat {
            val blurredMat = Mat()
            Imgproc.GaussianBlur(image, blurredMat, Size(5.0, 5.0), 0.0)
            return blurredMat
        }

        /**
         * Applies a morphological closing operation to fill small holes in the binary image.
         */
        fun applyMorphologicalClosing(image: Mat): Mat {
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
            val closedImage = Mat()
            Imgproc.morphologyEx(image, closedImage, Imgproc.MORPH_CLOSE, kernel)
            return closedImage
        }
    }
}

/**
 * Utility class for image conversions between Bitmap and Mat.
 */
class ImageUtils {
    companion object {
        /**
         * Converts a Bitmap to an OpenCV Mat.
         *
         * @param bitmap The input Bitmap.
         * @return The corresponding Mat.
         */
        fun bitmapToMat(bitmap: Bitmap): Mat = Mat().also {
            Utils.bitmapToMat(bitmap, it)
        }

        /**
         * Converts an OpenCV Mat to a Bitmap.
         *
         * @param mat The input Mat.
         * @return The corresponding Bitmap.
         */
        fun matToBitmap(mat: Mat): Bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888).apply {
            Utils.matToBitmap(mat, this)
        }
    }
}
