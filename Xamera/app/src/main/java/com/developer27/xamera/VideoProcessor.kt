package com.developer27.xamera

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

import org.opencv.core.*
import org.opencv.imgproc.*
import org.opencv.videoio.*

import java.io.File

interface VideoProcessorCallback {
    fun onVideoProcessed(videoUri: Uri)
}

class VideoProcessor(
    private val context: Context,
    private val callback: VideoProcessorCallback
) {

    init {
        initializeOpenCV()
    }

    private fun initializeOpenCV() {
        Log.i("VideoProcessor", "OpenCV should be loaded automatically with included libraries.")
        Toast.makeText(context, "OpenCV loaded successfully", Toast.LENGTH_SHORT).show()
    }

    fun processVideoWithOpenCV(videoUri: Uri) {
        notifyUser("Processing video...")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val tempVideoFile = createTempVideoFile(videoUri)
                val processedVideoFile = createOutputVideoFile()

                processAndWriteFrames(tempVideoFile, processedVideoFile)

                saveProcessedVideo(processedVideoFile)
            } catch (e: Exception) {
                handleProcessingError(e)
            }
        }
    }

    private fun createTempVideoFile(videoUri: Uri): File {
        val tempFile = File.createTempFile("temp_video", ".mp4", context.cacheDir)
        context.contentResolver.openInputStream(videoUri)?.use { input ->
            tempFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        return tempFile
    }

    private fun createOutputVideoFile(): File {
        return File(context.getExternalFilesDir(Environment.DIRECTORY_MOVIES),
            "processed_video_${System.currentTimeMillis()}.mp4")
    }

    private fun processAndWriteFrames(inputFile: File, outputFile: File) {
        val capture = VideoCapture(inputFile.absolutePath)
        if (!capture.isOpened) {
            notifyUser("Failed to open video")
            return
        }

        val fps = capture.get(Videoio.CAP_PROP_FPS)
        val frameWidth = capture.get(Videoio.CAP_PROP_FRAME_WIDTH).toInt()
        val frameHeight = capture.get(Videoio.CAP_PROP_FRAME_HEIGHT).toInt()
        val codec = VideoWriter.fourcc('H', '2', '6', '4')

        val writer = VideoWriter(outputFile.absolutePath, codec, fps, Size(frameWidth.toDouble(), frameHeight.toDouble()))
        if (!writer.isOpened) {
            notifyUser("Failed to open VideoWriter")
            capture.release()
            return
        }

        processEachFrame(capture, writer)
        capture.release()
        writer.release()
        inputFile.delete()
    }

    private fun processEachFrame(capture: VideoCapture, writer: VideoWriter) {
        val frame = Mat()
        val centerDataList = mutableListOf<Point>()

        while (capture.read(frame)) {
            val grayFrame = applyGrayscale(frame)
            val enhancedFrame = enhanceBrightness(grayFrame)
            val (center, overlayedFrame) = detectContourBlob(enhancedFrame)

            center?.let {
                centerDataList.add(it)
                drawTraceLine(overlayedFrame, centerDataList)
            }
            writer.write(overlayedFrame)
        }
    }

    private fun applyGrayscale(frame: Mat): Mat {
        val grayFrame = Mat()
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY)
        return grayFrame
    }

    private fun enhanceBrightness(image: Mat): Mat {
        val enhancedImage = Mat()
        Core.multiply(image, Scalar(2.0), enhancedImage)
        return enhancedImage
    }

    private fun detectContourBlob(image: Mat): Pair<Point?, Mat> {
        val binaryImage = Mat()
        Imgproc.threshold(image, binaryImage, 200.0, 255.0, Imgproc.THRESH_BINARY)

        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(binaryImage, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val largestContour = contours.maxByOrNull { Imgproc.contourArea(it) }
        val outputImage = Mat()
        Imgproc.cvtColor(image, outputImage, Imgproc.COLOR_GRAY2BGR)

        var center: Point? = null
        largestContour?.takeIf { Imgproc.contourArea(it) > 500 }?.let {
            Imgproc.drawContours(outputImage, listOf(it), -1, Scalar(255.0, 105.0, 180.0), Imgproc.FILLED)
            val moments = Imgproc.moments(it)
            center = calculateCenter(moments, outputImage)
        }
        return Pair(center, outputImage)
    }

    private fun calculateCenter(moments: Moments, image: Mat): Point? {
        return if (moments.m00 != 0.0) {
            val centerX = (moments.m10 / moments.m00).toInt()
            val centerY = (moments.m01 / moments.m00).toInt()
            val center = Point(centerX.toDouble(), centerY.toDouble())
            Imgproc.circle(image, center, 10, Scalar(0.0, 255.0, 0.0), -1)
            center
        } else null
    }

    private fun drawTraceLine(image: Mat, centerDataList: List<Point>) {
        for (i in 1 until centerDataList.size) {
            Imgproc.line(
                image,
                centerDataList[i - 1],
                centerDataList[i],
                Scalar(255.0, 0.0, 0.0),
                2
            )
        }
    }

    private fun saveProcessedVideo(processedVideoFile: File) {
        val videoUri = saveVideoToMediaStore(processedVideoFile)
        videoUri?.let {
            notifyUser("Processed video saved: $it")
            callback.onVideoProcessed(it)
        }
    }

    private fun saveVideoToMediaStore(file: File): Uri? {
        val contentValues = createMediaStoreContentValues(file.name)
        val resolver = context.contentResolver

        return resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, contentValues)?.also { videoUri ->
            resolver.openOutputStream(videoUri).use { outputStream ->
                file.inputStream().use { inputStream ->
                    inputStream.copyTo(outputStream!!)
                }
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                contentValues.clear()
                contentValues.put(MediaStore.MediaColumns.IS_PENDING, 0)
                resolver.update(videoUri, contentValues, null, null)
            }
            file.delete()
        }
    }

    private fun createMediaStoreContentValues(filename: String): ContentValues {
        return ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/Xamera-Processed")
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
        }
    }

    private fun handleProcessingError(e: Exception) {
        e.printStackTrace()
        notifyUser("Error processing video: ${e.message}")
    }

    private fun notifyUser(message: String) {
        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
    }
}