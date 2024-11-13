package com.developer27.xamera

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import kotlinx.coroutines.*
import org.opencv.core.*
import org.opencv.imgproc.*
import org.opencv.videoio.*
import java.io.File

// Callback interface to notify when video processing is complete
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

    // Initializes OpenCV; assumes libraries are bundled with the app
    private fun initializeOpenCV() {
        Log.i("VideoProcessor", "Loading OpenCV libraries.")
        Toast.makeText(context, "OpenCV loaded successfully", Toast.LENGTH_SHORT).show()
    }

    // Main function to process video using OpenCV
    fun processVideoWithOpenCV(videoUri: Uri) {
        notifyUser("Processing video...")

        CoroutineScope(Dispatchers.IO).launch {  // Run in a background thread
            try {
                // Step 1: Copy input video to a temporary file
                val tempVideoFile = createTempVideoFile(videoUri)

                // Step 2: Prepare the output file for the processed video
                val processedVideoFile = createOutputVideoFile()

                // Step 3: Process frames from the input video and write to output file
                processAndWriteFrames(tempVideoFile, processedVideoFile)

                // Step 4: Save the processed video to MediaStore
                saveProcessedVideo(processedVideoFile)
            } catch (e: Exception) {
                handleProcessingError(e)  // Handle any errors during processing
            }
        }
    }

    // Creates a temporary file from the video Uri for easier access
    private fun createTempVideoFile(videoUri: Uri): File {
        val tempFile = File.createTempFile("temp_video", ".mp4", context.cacheDir)
        context.contentResolver.openInputStream(videoUri)?.use { input ->
            tempFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        return tempFile
    }

    // Generates an output file in the app's external movies directory
    private fun createOutputVideoFile(): File {
        return File(context.getExternalFilesDir(Environment.DIRECTORY_MOVIES),
            "processed_video_${System.currentTimeMillis()}.mp4")
    }

    // Reads and processes each frame, then writes to the output video file
    private fun processAndWriteFrames(inputFile: File, outputFile: File) {
        val capture = VideoCapture(inputFile.absolutePath)
        if (!capture.isOpened) {
            notifyUser("Failed to open video")
            return
        }

        // Extract video properties for writing
        val fps = capture.get(Videoio.CAP_PROP_FPS)
        val frameWidth = capture.get(Videoio.CAP_PROP_FRAME_WIDTH).toInt()
        val frameHeight = capture.get(Videoio.CAP_PROP_FRAME_HEIGHT).toInt()
        val codec = VideoWriter.fourcc('H', '2', '6', '4')

        // Initialize VideoWriter with extracted properties
        val writer = VideoWriter(outputFile.absolutePath, codec, fps, Size(frameWidth.toDouble(), frameHeight.toDouble()))
        if (!writer.isOpened) {
            notifyUser("Failed to open VideoWriter")
            capture.release()
            return
        }

        // Process each frame and write it to output
        processEachFrame(capture, writer)

        // Release resources after processing
        capture.release()
        writer.release()
        inputFile.delete()
    }

    // Reads each frame, applies processing, and writes it to the writer
    private fun processEachFrame(capture: VideoCapture, writer: VideoWriter) {
        val frame = Mat()
        val centerDataList = mutableListOf<Point>()

        while (capture.read(frame)) {
            val grayFrame = applyGrayscale(frame)          // Convert to grayscale
            val enhancedFrame = enhanceBrightness(grayFrame)  // Enhance brightness
            val (center, overlayedFrame) = detectContourBlob(enhancedFrame)  // Detect contours

            // Draw trace line for detected center points
            center?.let {
                centerDataList.add(it)
                drawTraceLine(overlayedFrame, centerDataList)
            }
            writer.write(overlayedFrame)  // Write the processed frame to output
        }
    }

    // Converts frame to grayscale for easier processing
    private fun applyGrayscale(frame: Mat): Mat {
        val grayFrame = Mat()
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY)
        return grayFrame
    }

    // Enhances brightness to make contour detection easier
    private fun enhanceBrightness(image: Mat): Mat {
        val enhancedImage = Mat()
        Core.multiply(image, Scalar(2.0), enhancedImage)
        return enhancedImage
    }

    // Detects the largest contour in the image, returns center point and modified image
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

    // Calculates center point of a contour based on its moments
    private fun calculateCenter(moments: Moments, image: Mat): Point? {
        return if (moments.m00 != 0.0) {
            val centerX = (moments.m10 / moments.m00).toInt()
            val centerY = (moments.m01 / moments.m00).toInt()
            val center = Point(centerX.toDouble(), centerY.toDouble())
            Imgproc.circle(image, center, 10, Scalar(0.0, 255.0, 0.0), -1)  // Mark center
            center
        } else null
    }

    // Draws a continuous trace line from detected center points
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

    // Saves the processed video to MediaStore for easy access
    private fun saveProcessedVideo(processedVideoFile: File) {
        val videoUri = saveVideoToMediaStore(processedVideoFile)
        videoUri?.let {
            notifyUser("Processed video saved: $it")
            callback.onVideoProcessed(it)
        }
    }

    // Adds the processed video file to MediaStore
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

    // Generates required MediaStore metadata for the video file
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

    // Handles errors during video processing
    private fun handleProcessingError(e: Exception) {
        e.printStackTrace()
        notifyUser("Error processing video: ${e.message}")
    }

    // Displays a message to the user
    private fun notifyUser(message: String) {
        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
    }
}