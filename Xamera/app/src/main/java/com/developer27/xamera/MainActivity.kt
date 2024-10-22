package com.developer27.xamera

import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.preference.PreferenceManager
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
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
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Inflate the layout and bind views
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

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
                    }
                }
            }
    }

    // Function to start the camera preview and set up video capture
    private fun startCamera() {
        // Obtain a camera provider instance asynchronously
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Get the camera provider once it's ready
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Build the camera preview use case
            val preview = Preview.Builder()
                .build()
                .also {
                    // Set the surface provider for the preview (the viewFinder)
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            // Build the video capture use case
            videoCapture = VideoCapture.withOutput(Recorder.Builder().build())

            // Apply flash setting when the camera is started
            applyFlashIfEnabled()

            try {
                // Unbind any use cases before rebinding
                cameraProvider.unbindAll()

                // Bind the camera lifecycle with the use cases: preview and video capture
                val camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, videoCapture
                )

                // Initialize camera control for zooming
                cameraControl = camera.cameraControl
                cameraInfo = camera.cameraInfo

                // Apply lighting mode after camera is ready
                applyLightingMode()

            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
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
                if (it.key in REQUIRED_PERMISSIONS && it.value == false)
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
        } else {
        }
    }

    private fun setHighLightMode() {
        val exposureRange = cameraInfo.exposureState.exposureCompensationRange
        if (-2 in exposureRange) {
            cameraControl.setExposureCompensationIndex(-2)
        } else {
        }
    }

    private fun setNormalMode() {
        val exposureRange = cameraInfo.exposureState.exposureCompensationRange
        if (0 in exposureRange) {
            cameraControl.setExposureCompensationIndex(0)
        } else {
        }
    }
}
