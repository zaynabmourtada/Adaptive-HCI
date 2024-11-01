package com.developer27.xamera

import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.preference.PreferenceManager
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraControl
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.core.content.ContextCompat
import com.developer27.xamera.databinding.ActivityMainBinding
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.opencv.android.OpenCVLoader
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext


class MainActivity : AppCompatActivity() {

    // Binding object to access views
    private lateinit var viewBinding: ActivityMainBinding

    // Object for capturing videos
    private var videoCapture: VideoCapture<Recorder>? = null

    // Keeps track of video recording session
    private var recording: Recording? = null

    // Executor for running camera operations on a background thread
    private lateinit var cameraExecutor: ExecutorService

    // Camera selector for front or back camera
    private var cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    private lateinit var sharedPreferences: SharedPreferences

    // Camera and zoom ratio variables
    private lateinit var cameraControl: CameraControl
    private lateinit var cameraInfo: CameraInfo
    private var zoomLevel = 1.0f
    private val maxZoom = 5.0f

    // Declare the ActivityResultLauncher at the top
    private lateinit var videoPickerLauncher: ActivityResultLauncher<Intent>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Inflate the layout and bind views
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        if (OpenCVLoader.initLocal()) {
            Log.i(TAG, "OpenCV loaded successfully")
        }
        else
        {
            Log.e(TAG, "OpenCV initialization failed!")
            (Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG)).show()
            return
       }

        // Request camera permissions. If granted, start the camera; otherwise, request permissions.
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }

        // Set up listener for the "Capture Video" button
        viewBinding.videoCaptureButton.setOnClickListener {
            captureVideo()
            applyFlashIfEnabled() // Apply flash setting if enabled
        }

        // Apply flash setting once on camera start
        applyFlashIfEnabled()

        // Set up listener for the "Switch Camera" button
        viewBinding.switchCameraButton.setOnClickListener {
            // Toggle between front and back camera
            cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
                CameraSelector.DEFAULT_FRONT_CAMERA
            } else {
                CameraSelector.DEFAULT_BACK_CAMERA
            }

            // Restart the camera to apply the new camera selector
            startCamera()
        }

        // Initialize the executor for camera operations
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Set up zoom controls
        setupZoomControls()

        // Start camera and apply lighting mode settings
        cameraExecutor = Executors.newSingleThreadExecutor()
        startCamera()

        // Set up listener for iconImageButton
        viewBinding.aboutButton.setOnClickListener {
            val intent = Intent(this, AboutXameraActivity::class.java)
            startActivity(intent)
        }

        // Load and apply the rolling shutter frequency setting

        // Initialize the camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Set up settings button to open the settings activity
        viewBinding.settingsButton.setOnClickListener {
            val intent = Intent(this, SettingsActivity::class.java)
            startActivity(intent)
        }

        // Register shared preferences listener to detect changes in settings
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        sharedPreferences.registerOnSharedPreferenceChangeListener { _, key ->
            if (key == "lighting_mode") {
                applyLightingMode() // Reapply lighting mode if changed
            }
        }

        // Start the camera
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }

        // Initialize the ActivityResultLauncher for video selection
        videoPickerLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == RESULT_OK) {
                val selectedVideoUri: Uri? = result.data?.data
                if (selectedVideoUri != null) {
                    processVideoWithOpenCV(selectedVideoUri)  // Pass Uri directly
                } else {
                    Toast.makeText(this, "No video selected", Toast.LENGTH_SHORT).show()
                }
            } else {
                Toast.makeText(this, "Video selection canceled", Toast.LENGTH_SHORT).show()
            }
        }

        // Set up the button click listener to open Google Photos
        viewBinding.processVideoButton.setOnClickListener {
            openVideoPicker()
        }
    }

    // Function to start video recording or stop if already recording
    private fun captureVideo() {
        val videoCapture = this.videoCapture ?: return

        // Disable the capture button while recording is starting
        viewBinding.videoCaptureButton.isEnabled = false

        // Disable switch camera button while recording
        viewBinding.switchCameraButton.isEnabled = false

        val curRecording = recording
        if (curRecording != null) {
            // Stop the current recording
            curRecording.stop()
            recording = null
            return
        }

        // Create a timestamped file name and content values for the video file
        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            // Set the relative path for saving the video (for Android versions above P)
            if (Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/Xamera-Videos")
            }
        }

        // Set up output options for video file
        val mediaStoreOutputOptions = MediaStoreOutputOptions
            .Builder(contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            .setContentValues(contentValues)
            .build()

        // Prepare and start a new recording session
        recording = videoCapture.output
            .prepareRecording(this, mediaStoreOutputOptions)
            .apply {
                // Enable audio if permission is granted
                if (ContextCompat.checkSelfPermission(
                        this@MainActivity,
                        Manifest.permission.RECORD_AUDIO
                    ) == PackageManager.PERMISSION_GRANTED
                ) {
                    withAudioEnabled()
                }
            }
            .start(ContextCompat.getMainExecutor(this)) { recordEvent ->
                when (recordEvent) {
                    // Handle recording start event
                    is VideoRecordEvent.Start -> {
                        viewBinding.videoCaptureButton.apply {
                            text = getString(R.string.stop_capture)
                            backgroundTintList = ColorStateList.valueOf(
                                ContextCompat.getColor(context, R.color.red)
                            )
                            //Change title title text to red
                            viewBinding.titleText.apply {
                                backgroundTintList = ColorStateList.valueOf(
                                    ContextCompat.getColor(context, R.color.red)
                                )
                            }
                            isEnabled = true
                        }
                    }
                    // Handle recording stop and finalize event
                    is VideoRecordEvent.Finalize -> {
                        if (!recordEvent.hasError()) {
                            val msg = "Video capture succeeded: ${recordEvent.outputResults.outputUri}"
                            Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                            Log.d(TAG, msg)
                        } else {
                            recording?.close()
                            recording = null
                            Log.e(TAG, "Video capture ends with error: ${recordEvent.error}")
                        }
                        viewBinding.videoCaptureButton.apply {
                            text = getString(R.string.start_capture)
                            backgroundTintList = ColorStateList.valueOf(
                                ContextCompat.getColor(context, R.color.blue)
                            )
                            //Change title title text to blue
                            viewBinding.titleText.apply {
                                backgroundTintList = ColorStateList.valueOf(
                                    ContextCompat.getColor(context, R.color.darkBlue)
                                )
                            }
                            isEnabled = true
                        }
                        // Re-enable switch camera button
                        viewBinding.switchCameraButton.isEnabled = true

                        // Turn off the flash after the recording is finalized
                        cameraControl.enableTorch(false)
                    }
                }
            }
    }

    // Function to start the camera preview and set up video capture
// Function to start the camera preview and set up video capture
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
            }

            videoCapture = VideoCapture.withOutput(Recorder.Builder().build())

            applyRollingShutterFrequency()

            try {
                cameraProvider.unbindAll()
                val camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, videoCapture)
                cameraControl = camera.cameraControl
                cameraInfo = camera.cameraInfo

                // Set initial zoom level to 1.0 (no zoom)
                zoomLevel = 1.0f
                cameraControl.setLinearZoom(zoomLevel / maxZoom)

            } catch (exc: Exception) {
                Toast.makeText(this, "Use case binding failed: ${exc.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun applyRollingShutterFrequency() {
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        val frequency = sharedPreferences.getString("rolling_shutter_frequency", "60") ?: "60"

        when (frequency) {
            "50" -> setRollingShutterFrequency(50)
            "60" -> setRollingShutterFrequency(60)
        }
    }

    private fun setRollingShutterFrequency(frequency: Int) {
        if (!::cameraControl.isInitialized) {
            Toast.makeText(this, "Camera is not initialized", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val exposureRange = cameraInfo.exposureState.exposureCompensationRange
            val exposureCompensationIndex = when (frequency) {
                50 -> 2  // Example index for 50Hz
                60 -> -2  // Example index for 60Hz
                else -> 0  // Default or normal
            }

            if (exposureCompensationIndex in exposureRange) {
                cameraControl.setExposureCompensationIndex(exposureCompensationIndex)
                Toast.makeText(this, "Rolling shutter frequency set to $frequency Hz", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Exposure compensation index out of range for frequency $frequency Hz", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Error setting rolling shutter frequency: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun setupZoomControls() {
        val handler = Handler(Looper.getMainLooper())

        // Zoom in on short press
        viewBinding.zoomInButton.setOnClickListener {
            zoomIn()
        }

        // Zoom out on short press
        viewBinding.zoomOutButton.setOnClickListener {
            zoomOut()
        }

        // Continuous zoom in on long press
        viewBinding.zoomInButton.setOnLongClickListener {
            handler.post(object : Runnable {
                override fun run() {
                    zoomIn()
                    handler.postDelayed(this, 100) // Repeat every 100ms
                }
            })
            true // Return true to indicate the event has been handled
        }

        // Continuous zoom out on long press
        viewBinding.zoomOutButton.setOnLongClickListener {
            handler.post(object : Runnable {
                override fun run() {
                    zoomOut()
                    handler.postDelayed(this, 100) // Repeat every 100ms
                }
            })
            true // Return true to indicate the event has been handled
        }

        // Stop zooming when the buttons are released
        viewBinding.zoomInButton.setOnTouchListener { _, event ->
            if (event.action == android.view.MotionEvent.ACTION_UP || event.action == android.view.MotionEvent.ACTION_CANCEL) {
                handler.removeCallbacksAndMessages(null) // Stop continuous zooming
            }
            false
        }

        viewBinding.zoomOutButton.setOnTouchListener { _, event ->
            if (event.action == android.view.MotionEvent.ACTION_UP || event.action == android.view.MotionEvent.ACTION_CANCEL) {
                handler.removeCallbacksAndMessages(null) // Stop continuous zooming
            }
            false
        }
    }

    // Function to apply flash if enabled
    private fun applyFlashIfEnabled() {
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        val isFlashEnabled = sharedPreferences.getBoolean("enable_flash", false)

        if (::cameraControl.isInitialized) {
            cameraControl.enableTorch(isFlashEnabled)  // Enable or disable flash
        }
    }

    // Function to zoom in
    private fun zoomIn() {
        if (zoomLevel < maxZoom) {
            zoomLevel += 0.1f
            cameraControl.setLinearZoom(zoomLevel / maxZoom)
        }
    }

    // Function to zoom out
    private fun zoomOut() {
        if (zoomLevel > 1.0f) {
            zoomLevel -= 0.1f
            cameraControl.setLinearZoom(zoomLevel / maxZoom)
        }
    }

    // Function to request camera and audio permissions
    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    // Check if all required permissions have been granted
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onResume() {
        super.onResume()
        // Reset zoom level to minimum (1.0) on resume
        if (::cameraControl.isInitialized) {
            zoomLevel = 1.0f
            cameraControl.setLinearZoom(zoomLevel / maxZoom)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Shut down the camera executor when activity is destroyed
        cameraExecutor.shutdown()
    }

    companion object {
        // Tag for logging
        private const val TAG = "Xamera"

        // File name format for videos
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"

        // List of required permissions
        private val REQUIRED_PERMISSIONS =
            mutableListOf(
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            ).apply {
                // Add WRITE_EXTERNAL_STORAGE permission for Android versions below P
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }

    // Activity result launcher for requesting multiple permissions
    private val activityResultLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            var permissionGranted = true
            // Check if all permissions are granted
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && !it.value)
                    permissionGranted = false
            }
            if (!permissionGranted) {
                // Show toast if permission request is denied
                Toast.makeText(baseContext, "Permission request denied", Toast.LENGTH_SHORT).show()
            } else {
                // Start the camera if permissions are granted
                startCamera()
            }
        }

    private fun applyLightingMode() {
        // Ensure cameraControl is initialized before applying settings
        if (!::cameraControl.isInitialized) {
            Toast.makeText(this, "Camera not initialized", Toast.LENGTH_SHORT).show()
            return
        }

        // Get the selected lighting mode from shared preferences
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        val lightingMode = sharedPreferences.getString("lighting_mode", "normal")

        // Adjust camera settings based on the lighting mode
        when (lightingMode) {
            "low_light" -> setLowLightMode()
            "high_light" -> setHighLightMode()
            else -> setNormalMode()
        }
    }

    private fun setLowLightMode() {
        // Ensure exposure adjustment is possible
        val exposureRange = cameraInfo.exposureState.exposureCompensationRange
        if (2 in exposureRange) {
            cameraControl.setExposureCompensationIndex(2)
        }
    }

    private fun setHighLightMode() {
        val exposureRange = cameraInfo.exposureState.exposureCompensationRange
        if (-2 in exposureRange) {
            cameraControl.setExposureCompensationIndex(-2)
        }
    }

    private fun setNormalMode() {
        val exposureRange = cameraInfo.exposureState.exposureCompensationRange
        if (0 in exposureRange) {
            cameraControl.setExposureCompensationIndex(0)
        }
    }

    //Pick a video from the file
    private fun openVideoPicker() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
        videoPickerLauncher.launch(intent)
    }

    // Function to process and save the video with OpenCV
    private fun processVideoWithOpenCV(videoUri: Uri) {
        Toast.makeText(this, "Video selected: $videoUri", Toast.LENGTH_SHORT).show()

        val fileName = "video-PreProcessed.mp4"
        val processedVideoFile = File(getExternalFilesDir(Environment.DIRECTORY_MOVIES), fileName)
        val videoPath = getRealPathFromUri(videoUri) ?: return

        val capture = org.opencv.videoio.VideoCapture(videoPath)
        if (!capture.isOpened) {
            Toast.makeText(this, "Failed to open video", Toast.LENGTH_SHORT).show()
            return
        }

        // Set up VideoWriter to save the processed video
        val fps = capture.get(org.opencv.videoio.Videoio.CAP_PROP_FPS)
        val frameWidth = capture.get(org.opencv.videoio.Videoio.CAP_PROP_FRAME_WIDTH).toInt()
        val frameHeight = capture.get(org.opencv.videoio.Videoio.CAP_PROP_FRAME_HEIGHT).toInt()
        val writer = org.opencv.videoio.VideoWriter(
            processedVideoFile.absolutePath,
            org.opencv.videoio.VideoWriter.fourcc('M', 'P', '4', 'V'),
            fps,
            org.opencv.core.Size(frameWidth.toDouble(), frameHeight.toDouble())
        )

        if (!writer.isOpened) {
            Toast.makeText(this, "Failed to open VideoWriter", Toast.LENGTH_SHORT).show()
            capture.release()
            return
        }

        // Process each frame and write it to the new video
        val frame = org.opencv.core.Mat()
        val centerDataList = mutableListOf<org.opencv.core.Point>() // For drawing continuous trace

        while (capture.read(frame)) {
            // Convert to grayscale
            val grayFrame = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.cvtColor(frame, grayFrame, org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY)

            // Enhance brightness
            val enhancedFrame = enhanceBrightness(grayFrame)

            // Detect contour blob and overlay center mass
            val contourData = detectContourBlob(enhancedFrame)
            val overlayedFrame = contourData.second

            // Draw continuous trace line for detected center points
            contourData.first?.let { center ->
                centerDataList.add(center)
                for (i in 1 until centerDataList.size) {
                    org.opencv.imgproc.Imgproc.line(
                        overlayedFrame,
                        centerDataList[i - 1],
                        centerDataList[i],
                        org.opencv.core.Scalar(255.0, 0.0, 0.0),
                        2
                    )
                }
            }
            // Write the processed frame to the output video
            writer.write(overlayedFrame)
        }
        // Release resources
        capture.release()
        writer.release()
        // Save the processed video to the gallery
        saveProcessedVideo(processedVideoFile)
    }

    // Helper function to get absolute path from Uri
    private fun getRealPathFromUri(uri: Uri): String? {
        var filePath: String? = null
        val projection = arrayOf(MediaStore.Video.Media.DATA)
        contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DATA)
                filePath = cursor.getString(columnIndex)
            }
        }
        return filePath
    }

    // Function to enhance brightness
    private fun enhanceBrightness(image: org.opencv.core.Mat): org.opencv.core.Mat {
        val enhancedImage = image.clone()
        org.opencv.core.Core.multiply(image, org.opencv.core.Scalar(2.0), enhancedImage)
        return enhancedImage
    }

    // Function to detect contour blob and find center
    private fun detectContourBlob(image: org.opencv.core.Mat): Pair<org.opencv.core.Point?, org.opencv.core.Mat> {
        val binaryImage = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.threshold(image, binaryImage, 200.0, 255.0, org.opencv.imgproc.Imgproc.THRESH_BINARY)
        val contours = mutableListOf<org.opencv.core.MatOfPoint>()
        org.opencv.imgproc.Imgproc.findContours(binaryImage, contours, org.opencv.core.Mat(), org.opencv.imgproc.Imgproc.RETR_EXTERNAL, org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE)
        var maxArea = 0.0
        var largestContour: org.opencv.core.MatOfPoint? = null
        for (contour in contours) {
            val area = org.opencv.imgproc.Imgproc.contourArea(contour)
            if (area > 500 && area > maxArea) {
                maxArea = area
                largestContour = contour
            }
        }
        val outputImage = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.cvtColor(image, outputImage, org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR)
        var center: org.opencv.core.Point? = null
        largestContour?.let {
            org.opencv.imgproc.Imgproc.drawContours(outputImage, listOf(it), -1, org.opencv.core.Scalar(255.0, 105.0, 180.0), org.opencv.core.Core.FILLED)
            val moments = org.opencv.imgproc.Imgproc.moments(it)
            if (moments.m00 != 0.0) {
                val centerX = (moments.m10 / moments.m00).toInt()
                val centerY = (moments.m01 / moments.m00).toInt()
                center = org.opencv.core.Point(centerX.toDouble(), centerY.toDouble())
                org.opencv.imgproc.Imgproc.circle(outputImage, center, 10, org.opencv.core.Scalar(0.0, 255.0, 0.0), -1)
            }
        }
        return Pair(center, outputImage)
    }

    // Save the video after it is processed
    private fun saveProcessedVideo(processedVideoFile: File) {
        val filename = "processed_video_${System.currentTimeMillis()}.mp4"
        val mimeType = "video/mp4"
        val directory = Environment.DIRECTORY_MOVIES + "/Xamera-Processed"
        val resolver = contentResolver
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
            put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.MediaColumns.RELATIVE_PATH, directory)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
        }
        val videoUri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, contentValues)
            ?: run {
                Toast.makeText(this, "Failed to create file", Toast.LENGTH_SHORT).show()
                return
            }
        resolver.openOutputStream(videoUri).use { outputStream ->
            processedVideoFile.inputStream().use { inputStream ->
                inputStream.copyTo(outputStream!!)
            }
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            contentValues.clear()
            contentValues.put(MediaStore.MediaColumns.IS_PENDING, 0)
            resolver.update(videoUri, contentValues, null, null)
        }
        Toast.makeText(this, "Video saved to gallery", Toast.LENGTH_SHORT).show()
    }
}
