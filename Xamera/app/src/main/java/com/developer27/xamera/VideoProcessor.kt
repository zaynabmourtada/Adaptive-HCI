// File: VideoProcessor.kt
package com.developer27.xamera

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

object Settings {
    object Contour {
        var threshold = 500
    }
    object Trace {
        var lineLimit = 50
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

data class FrameData(val x: Double, val y: Double, val area: Double, val frameCount: Int)

class VideoProcessor(private val context: Context) {

    private lateinit var kalmanFilter: KalmanFilter
    private val centerDataList = LinkedList<Point>() // Real-time trace with limited points
    private val preFilter4Ddata = mutableListOf<FrameData>() // Complete raw data
    private val postFilter4Ddata = mutableListOf<FrameData>() // Complete filtered data
    private var frameCount = 0

    init {
        initializeOpenCV()
        //System.loadLibrary("opencv_java3")
    }

    private fun initializeOpenCV() {
        if (OpenCVLoader.initDebug()) {
            showToast("OpenCV loaded successfully")
            initializeKalmanFilter()
        } else {
            //showToast("OpenCV initialization failed!")
        }
    }

    private fun initializeKalmanFilter() {
        kalmanFilter = KalmanFilter(4, 2)
        kalmanFilter._transitionMatrix = Mat.eye(4, 4, CvType.CV_32F).apply {
            put(0, 2, 1.0) // Linking position to velocity for x
            put(1, 3, 1.0) // Linking position to velocity for y
        }
        kalmanFilter._measurementMatrix = Mat.eye(2, 4, CvType.CV_32F)
        kalmanFilter._processNoiseCov = Mat.eye(4, 4, CvType.CV_32F).apply { setTo(Scalar(1e-4)) }
        kalmanFilter._measurementNoiseCov = Mat.eye(2, 2, CvType.CV_32F).apply { setTo(Scalar(1e-2)) }
        kalmanFilter._errorCovPost = Mat.eye(4, 4, CvType.CV_32F)
        //showToast("Kalman filter initialized")
    }

    fun clearTrackingData() {
        frameCount = 0
        preFilter4Ddata.clear()
        postFilter4Ddata.clear()
        centerDataList.clear()
        showToast("Tracking started: data reset.")
    }

    suspend fun processFrame(bitmap: Bitmap): Bitmap? = withContext(Dispatchers.Default) {
        try {
            val mat = ImageUtils.bitmapToMat(bitmap)

            val grayMat = Preprocessing.applyGrayscale(mat).also { mat.release() }
            val enhancedMat = Preprocessing.enhanceBrightness(grayMat).also { grayMat.release() }
            val thresholdMat = Preprocessing.conditionalThresholding(enhancedMat).also { enhancedMat.release() }
            val blurredMat = Preprocessing.applyGaussianBlur(thresholdMat).also { thresholdMat.release() }
            val cleanedMat = Preprocessing.applyMorphologicalClosing(blurredMat).also { blurredMat.release() }

            val (centerInfo, processedMat) = detectContourBlob(cleanedMat).also { cleanedMat.release() }
            val (center, area) = centerInfo

            center?.let {
                // Store raw point in preFilter4Ddata
                val frameData = FrameData(it.x, it.y, area ?: 0.0, frameCount++)
                preFilter4Ddata.add(frameData)
                // Apply Kalman filter to smooth the data and store in postFilter4Ddata
                applyKalmanFilter(it, area, frameData.frameCount)
                // Update trace for real-time display
                updateCenterTrace(it, processedMat)
                //logDebug("Raw Data:  | Frame(T)=${frameData.frameCount} | X=${frameData.x} | Y=${frameData.y} | Area=${frameData.area}")
            }

            return@withContext ImageUtils.matToBitmap(processedMat).also { processedMat.release() }
        } catch (e: Exception) {
            Log.e("VideoProcessor", "Error processing frame: ${e.message}")
            e.printStackTrace()
            return@withContext null
        }
    }

    private fun applyKalmanFilter(center: Point, area: Double?, frame: Int) {
        val measurement = Mat(2, 1, CvType.CV_32F).apply {
            put(0, 0, center.x)
            put(1, 0, center.y)
        }

        kalmanFilter.predict()

        val corrected = kalmanFilter.correct(measurement)
        val filteredX = corrected[0, 0][0]
        val filteredY = corrected[1, 0][0]

        // Store filtered data in postFilter4Ddata
        val filteredFrameData = FrameData(filteredX, filteredY, area ?: 0.0, frame)
        postFilter4Ddata.add(filteredFrameData)
        //logDebug("Filtered Data: | Frame(T)=${filteredFrameData.frameCount} | X=${filteredFrameData.x} | Y=${filteredFrameData.y} | Area=${filteredFrameData.area}")
    }

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

    private fun calculateCenter(contour: MatOfPoint, image: Mat): Pair<Point?, Pair<Int, Int>?> {
        val moments = Imgproc.moments(contour)
        return if (moments.m00 != 0.0) {
            val centerX = (moments.m10 / moments.m00).toInt()
            val centerY = (moments.m01 / moments.m00).toInt()
            val centerPoint = Point(centerX.toDouble(), centerY.toDouble())
            Imgproc.circle(image, centerPoint, 10, Scalar(0.0, 255.0, 0.0), -1)
            Pair(centerPoint, Pair(centerX, centerY))
        } else Pair(null, null)
    }

    private fun updateCenterTrace(center: Point, image: Mat) {
        centerDataList.add(center)
        if (centerDataList.size > Settings.Trace.lineLimit) centerDataList.removeFirst()
        for (i in 1 until centerDataList.size) {
            Imgproc.line(image, centerDataList[i - 1], centerDataList[i], Scalar(255.0, 0.0, 0.0), 2)
        }
    }

    fun retrievePreFilter4Ddata(): List<FrameData> = preFilter4Ddata
    fun retrievePostFilter4Ddata(): List<FrameData> = postFilter4Ddata

    private fun showToast(message: String) {
        if (Settings.Debug.enableToasts) Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
    }

    private fun logDebug(message: String) {
        if (Settings.Debug.enableLogging) Log.d("VideoProcessor", message)
    }
}

/**
 * Preprocessing utilities for image processing.
 */
class Preprocessing {
    companion object {
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
}

/**
 * Utility class for image conversion between Bitmap and Mat.
 */
class ImageUtils {
    companion object {
        /**
         * Converts a Bitmap to an OpenCV Mat.
         * @param bitmap The input Bitmap.
         * @return The corresponding Mat.
         */
        fun bitmapToMat(bitmap: Bitmap): Mat = Mat().also {
            org.opencv.android.Utils.bitmapToMat(bitmap, it)
        }

        /**
         * Converts an OpenCV Mat to a Bitmap.
         * @param mat The input Mat.
         * @return The corresponding Bitmap.
         */
        fun matToBitmap(mat: Mat): Bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888).apply {
            org.opencv.android.Utils.matToBitmap(mat, this)
        }
    }
}
