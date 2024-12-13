package com.developer27.xamera

import android.graphics.Bitmap
import android.content.Context
import android.util.Log
import android.widget.Toast
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator
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
        var lineLimit = Int.MAX_VALUE // We never remove old points, ensuring a continuous line
        var splineStep = 0.01
        // Light Blue (Cyan) line for the spline: B=255, G=255, R=0 in BGR
        var splineLineColor = Scalar(255.0, 255.0, 0.0)
        var lineThickness = 2
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

/**
 * VideoProcessor:
 * - Detects contours and finds their center using OpenCV.
 * - Applies a Kalman filter to smooth the detected center points, reducing jitter from tremors.
 * - Uses spline interpolation on the filtered points to draw a smooth curve.
 * - Draws only a light blue spline line and a red center dot. The line never disappears until "Stop Tracking" is pressed.
 * - Employs no removal of old points, ensuring continuous accumulation of the gesture's history.
 *
 * TODO <Zaynab Mourtada>: Integrate PyTorch inference if needed, possibly to classify gesture patterns.
 */
class VideoProcessor(private val context: Context) {

    private lateinit var kalmanFilter: KalmanFilter

    // Stores all filtered points without clearing, enabling a continuous, growing line
    private val centerDataList = LinkedList<Point>()

    private val preFilter4Ddata = mutableListOf<FrameData>()
    private val postFilter4Ddata = mutableListOf<FrameData>()

    private var frameCount = 0

    init {
        initializeOpenCV()
    }

    private fun initializeOpenCV() {
        if (OpenCVLoader.initDebug()) {
            showToast("OpenCV loaded successfully")
            initializeKalmanFilter()
        } else {
            // If OpenCV initialization fails, handle it if necessary
        }
    }

    private fun initializeKalmanFilter() {
        kalmanFilter = KalmanFilter(4, 2)
        kalmanFilter._transitionMatrix = Mat.eye(4,4,CvType.CV_32F).apply {
            put(0,2,1.0) // Link position to velocity for x
            put(1,3,1.0) // Link position to velocity for y
        }
        kalmanFilter._measurementMatrix = Mat.eye(2,4,CvType.CV_32F)
        kalmanFilter._processNoiseCov = Mat.eye(4,4,CvType.CV_32F).apply { setTo(Scalar(1e-4)) }
        kalmanFilter._measurementNoiseCov = Mat.eye(2,2,CvType.CV_32F).apply { setTo(Scalar(1e-2)) }
        kalmanFilter._errorCovPost = Mat.eye(4,4,CvType.CV_32F)
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
                val frameData = FrameData(it.x, it.y, area ?: 0.0, frameCount++)
                preFilter4Ddata.add(frameData)

                // Apply Kalman filter to get filtered coordinates
                val (filteredX, filteredY) = applyKalmanFilter(it, area, frameData.frameCount)

                // Use filtered coordinates for smoothing and line drawing
                val filteredPoint = Point(filteredX, filteredY)
                updateCenterTrace(filteredPoint, processedMat)
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
                Imgproc.drawContours(outputImage, listOf(it), -1, Scalar(255.0,105.0,180.0), Imgproc.FILLED)
                val area = Imgproc.contourArea(it)
                val (center, _) = calculateCenter(it, outputImage)
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

            // Draw center in red: B=0,G=0,R=255
            Imgproc.circle(image, centerPoint, 10, Scalar(0.0,0.0,255.0), -1)
            Pair(centerPoint, Pair(centerX, centerY))
        } else Pair(null, null)
    }

    /**
     * Draws a spline curve through all filtered points in centerDataList.
     */
    private fun drawSplineCurve(data: List<Point>, image: Mat) {
        if (data.size < 2) return

        val interpolator = SplineInterpolator()
        val xData = data.map { it.x }.toDoubleArray()
        val yData = data.map { it.y }.toDoubleArray()
        val tData = data.indices.map { it.toDouble() }.toDoubleArray()

        val splineX = interpolator.interpolate(tData, xData)
        val splineY = interpolator.interpolate(tData, yData)

        var prevPoint: Point? = null
        var t = 0.0
        while (t <= (data.size - 1).toDouble()) {
            val interpolatedX = splineX.value(t)
            val interpolatedY = splineY.value(t)
            val currentPoint = Point(interpolatedX, interpolatedY)

            prevPoint?.let {
                // Draw the spline in light blue (cyan)
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

    /**
     * Updates the trace by adding the new filtered point and drawing the spline curve.
     */
    private fun updateCenterTrace(filteredPoint: Point, image: Mat) {
        centerDataList.add(filteredPoint)
        drawSplineCurve(centerDataList, image)
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

class Preprocessing {
    companion object {
        fun applyGrayscale(frame: Mat): Mat {
            val grayMat = Mat()
            Imgproc.cvtColor(frame,grayMat,Imgproc.COLOR_BGR2GRAY)
            return grayMat
        }

        fun enhanceBrightness(image: Mat): Mat = Mat().apply {
            Core.multiply(image, Scalar(Settings.Brightness.factor), this)
        }

        fun conditionalThresholding(image: Mat): Mat {
            val thresholdMat = Mat()
            Imgproc.threshold(image,thresholdMat,Settings.Brightness.threshold,255.0,Imgproc.THRESH_TOZERO)
            return thresholdMat
        }

        fun applyGaussianBlur(image: Mat): Mat {
            val blurredMat = Mat()
            Imgproc.GaussianBlur(image,blurredMat,Size(5.0,5.0),0.0)
            return blurredMat
        }

        fun applyMorphologicalClosing(image: Mat): Mat {
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,Size(3.0,3.0))
            val closedImage = Mat()
            Imgproc.morphologyEx(image,closedImage,Imgproc.MORPH_CLOSE,kernel)
            return closedImage
        }
    }
}

class ImageUtils {
    companion object {
        fun bitmapToMat(bitmap: Bitmap): Mat = Mat().also {
            org.opencv.android.Utils.bitmapToMat(bitmap,it)
        }

        fun matToBitmap(mat: Mat): Bitmap = Bitmap.createBitmap(mat.cols(),mat.rows(),Bitmap.Config.ARGB_8888).apply {
            org.opencv.android.Utils.matToBitmap(mat,this)
        }
    }
}
