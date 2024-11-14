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
import java.util.*

class VideoProcessor(private val context: Context) {

    private val centerDataList = LinkedList<Point>()  // Use LinkedList for efficient queue management

    init {
        initializeOpenCV()
    }

    // Initializes OpenCV; assumes libraries are bundled with the app
    private fun initializeOpenCV() {
        if (OpenCVLoader.initDebug()) {
            Toast.makeText(context, "OpenCV loaded successfully", Toast.LENGTH_LONG).show()
        } else {
            Toast.makeText(context, "OpenCV initialization failed!", Toast.LENGTH_LONG).show()
        }
    }

    // Main function to process each frame; converts, enhances, detects and returns the processed bitmap
    suspend fun processFrame(bitmap: Bitmap): Bitmap? = withContext(Dispatchers.Default) {
        try {
            val mat = bitmapToMat(bitmap)
            val enhancedMat = enhanceBrightness(applyGrayscale(mat))
            val (center, processedMat) = detectContourBlob(enhancedMat)
            center?.let { updateCenterTrace(it, processedMat) }
            val processedBitmap = matToBitmap(processedMat)

            // Release resources
            mat.release()
            enhancedMat.release()
            processedMat.release()

            return@withContext processedBitmap
        } catch (e: Exception) {
            Log.e("VideoProcessor", "Error processing frame: ${e.message}")
            e.printStackTrace()
            return@withContext null
        }
    }

    // Converts bitmap to Mat for OpenCV processing
    private fun bitmapToMat(bitmap: Bitmap): Mat {
        val mat = Mat()
        org.opencv.android.Utils.bitmapToMat(bitmap, mat)
        return mat
    }

    // Converts processed Mat back to bitmap for display
    private fun matToBitmap(mat: Mat): Bitmap {
        return Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888).apply {
            org.opencv.android.Utils.matToBitmap(mat, this)
        }
    }

    // Converts frame to grayscale for easier processing
    private fun applyGrayscale(frame: Mat): Mat {
        val grayMat = Mat()
        Imgproc.cvtColor(frame, grayMat, Imgproc.COLOR_BGR2GRAY)
        return grayMat
    }

    // Enhances brightness for easier contour detection
    private fun enhanceBrightness(image: Mat): Mat {
        val enhancedImage = Mat()
        Core.multiply(image, Scalar(2.0), enhancedImage)
        return enhancedImage
    }

    // Detects the largest contour in the image, finds center point, and returns it along with processed Mat
    private fun detectContourBlob(image: Mat): Pair<Point?, Mat> {
        val binaryImage = Mat()
        Imgproc.threshold(image, binaryImage, 200.0, 255.0, Imgproc.THRESH_BINARY)

        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(binaryImage, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val largestContour = contours.maxByOrNull { Imgproc.contourArea(it) }
        val outputImage = Mat()
        Imgproc.cvtColor(image, outputImage, Imgproc.COLOR_GRAY2BGR)

        val center = largestContour?.takeIf { Imgproc.contourArea(it) > 500 }?.let {
            Imgproc.drawContours(outputImage, listOf(it), -1, Scalar(255.0, 105.0, 180.0), Imgproc.FILLED)
            calculateCenter(it, outputImage)
        }
        return Pair(center, outputImage)
    }

    // Calculates the center point of a contour based on its moments and draws it on the image
    private fun calculateCenter(contour: MatOfPoint, image: Mat): Point? {
        val moments = Imgproc.moments(contour)
        return if (moments.m00 != 0.0) {
            val centerX = (moments.m10 / moments.m00).toInt()
            val centerY = (moments.m01 / moments.m00).toInt()
            Point(centerX.toDouble(), centerY.toDouble()).also { center ->
                Imgproc.circle(image, center, 10, Scalar(0.0, 255.0, 0.0), -1)
            }
        } else null
    }

    // Updates the centerDataList for trace line management, and draws the trace line on the image
    private fun updateCenterTrace(center: Point, image: Mat) {
        centerDataList.add(center)
        if (centerDataList.size > 50) centerDataList.removeFirst()  // Keep list size manageable

        for (i in 1 until centerDataList.size) {
            Imgproc.line(image, centerDataList[i - 1], centerDataList[i], Scalar(255.0, 0.0, 0.0), 2)
        }
    }
}