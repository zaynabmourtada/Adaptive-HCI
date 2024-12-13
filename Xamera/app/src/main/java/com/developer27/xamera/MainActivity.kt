package com.developer27.xamera

// TODO <Zaynab Mourtada>: Import PyTorch libraries if needed
// import org.pytorch.IValue
// import org.pytorch.Module
// import org.pytorch.Tensor
// import org.pytorch.torchvision.TensorImageUtils

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Rect
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.preference.PreferenceManager
import android.util.Log
import android.util.Size
import android.util.SparseIntArray
import android.view.MotionEvent
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresPermission
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.developer27.xamera.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    // View binding for easier access to UI elements
    private lateinit var viewBinding: ActivityMainBinding

    // SharedPreferences for user settings (e.g., shutter speed)
    private lateinit var sharedPreferences: SharedPreferences

    // Camera-related fields
    private lateinit var cameraManager: CameraManager
    private var cameraDevice: CameraDevice? = null
    private lateinit var cameraId: String
    private var previewSize: Size? = null
    private var videoSize: Size? = null
    private var cameraCaptureSessions: CameraCaptureSession? = null
    private var captureRequestBuilder: CaptureRequest.Builder? = null

    // Flag to track if we are in "tracking" mode
    private var isTracking = false

    // Background thread and handler for camera operations
    private var backgroundHandler: Handler? = null
    private var backgroundThread: HandlerThread? = null

    // Zoom-related fields
    private var zoomLevel = 1.0f
    private val maxZoom = 10.0f
    private var sensorArraySize: Rect? = null

    // Permissions needed at runtime
    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )

    // ActivityResultLauncher for handling runtime permissions
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    // TextureView for camera preview
    private lateinit var textureView: TextureView

    // VideoProcessor for processing frames (drawing lines via OpenCV)
    private lateinit var videoProcessor: VideoProcessor

    // Flag to indicate which camera is currently active (front/back)
    private var isFrontCamera = false

    // Shutter speed control
    private var shutterSpeed: Long = 1000000000L / 60

    // Flag to prevent overlapping frame processing
    private var isProcessingFrame = false

    // TODO <Zaynab Mourtada>: Handle PyTorch model loading and inference
    // private var pytorchModule: Module? = null

    // TODO <Zaynab Mourtada>: Any PyTorch-related code should be implemented by Zaynab
    // For now, PyTorch references are commented out and replaced by TODO comments.

    // OpenGLTextureView for OpenGL rendering is turned off now; never made visible
    private lateinit var glTextureView: OpenGLTextureView

    // Listener for TextureView (camera preview lifecycle)
    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            // When TextureView is ready and permissions granted, open camera
            if (allPermissionsGranted()) {
                openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        }

        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean { return false }
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            // Called for every new camera frame.
            // If tracking is active, process the frame
            if (isTracking) {
                processFrameWithVideoProcessor()
            }
        }
    }

    // CameraDevice StateCallback to handle camera open/close/error
    private val stateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            cameraDevice = camera
            createCameraPreview()
        }
        override fun onDisconnected(camera: CameraDevice) {
            cameraDevice?.close()
            cameraDevice = null
        }
        override fun onError(camera: CameraDevice, error: Int) {
            cameraDevice?.close()
            cameraDevice = null
            finish()
        }
    }

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Inflate layout and bind UI
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Hide processed frame view until tracking starts
        viewBinding.processedFrameView.visibility = View.GONE

        // Initialize SharedPreferences
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)

        // Listen for changes in SharedPreferences (e.g., shutter speed changes)
        sharedPreferences.registerOnSharedPreferenceChangeListener { prefs, key ->
            if (key == "shutter_speed") {
                val shutterSpeedSetting = prefs.getString("shutter_speed", "60")?.toInt() ?: 60
                shutterSpeed = 1000000000L / shutterSpeedSetting
                updateShutterSpeed()
            }
        }

        // Load initial shutter speed
        val shutterSpeedSetting = sharedPreferences.getString("shutter_speed", "60")?.toInt() ?: 60
        shutterSpeed = 1000000000L / shutterSpeedSetting

        // Initialize CameraManager
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

        // The TextureView that shows camera preview
        textureView = viewBinding.viewFinder

        // Initialize the VideoProcessor
        videoProcessor = VideoProcessor(this)

        // OpenGLTextureView is present but never made visible (OpenGL "off")
        glTextureView = viewBinding.glTextureView

        // Initialize permission launcher for runtime permissions
        requestPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            val cameraPermissionGranted = permissions[Manifest.permission.CAMERA] ?: false
            val audioPermissionGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false

            if (cameraPermissionGranted && audioPermissionGranted) {
                if (textureView.isAvailable) {
                    openCamera()
                } else {
                    textureView.surfaceTextureListener = textureListener
                }
            } else {
                Toast.makeText(this, "Camera and Audio permissions are required.", Toast.LENGTH_SHORT).show()
            }
        }

        // If permissions already granted, open camera; else request
        if (allPermissionsGranted()) {
            if (textureView.isAvailable) {
                openCamera()
            } else {
                textureView.surfaceTextureListener = textureListener
            }
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        // Start/Stop Tracking button logic
        viewBinding.startTrackingButton.setOnClickListener {
            if (isTracking) {
                // If currently tracking, stop it and clear data
                stopTracking()
            } else {
                // Start tracking: show processed frame
                isTracking = true
                viewBinding.startTrackingButton.text = "Stop Tracking"
                viewBinding.startTrackingButton.backgroundTintList =
                    ContextCompat.getColorStateList(this, R.color.red)
                viewBinding.processedFrameView.visibility = View.VISIBLE

                // Since OpenGL is off, we do not show glTextureView
                videoProcessor.clearTrackingData()
            }
        }

        // Switch camera button
        viewBinding.switchCameraButton.setOnClickListener {
            switchCamera()
        }

        // Setup zoom controls
        setupZoomControls()

        // About button
        viewBinding.aboutButton.setOnClickListener {
            val intent = Intent(this, AboutXameraActivity::class.java)
            startActivity(intent)
        }

        // Settings button
        viewBinding.settingsButton.setOnClickListener {
            val intent = Intent(this, SettingsActivity::class.java)
            startActivityForResult(intent, SETTINGS_REQUEST_CODE)
        }

        // TODO <Zaynab Mourtada>: Load PyTorch model if needed
        // loadPyTorchModel()
    }

    // TODO <Zaynab Mourtada>: Implement this function to load the correct PyTorch model
    // private fun loadPyTorchModel() {
    //     // TODO <Zaynab Mourtada>: Load PyTorch model here
    // }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == SETTINGS_REQUEST_CODE && resultCode == RESULT_OK) {
            val shutterSpeedSetting = sharedPreferences.getString("shutter_speed", "60")?.toInt() ?: 60
            shutterSpeed = 1000000000L / shutterSpeedSetting
            Toast.makeText(this, "Shutter speed updated", Toast.LENGTH_SHORT).show()
            updateShutterSpeed()
        }
    }

    @SuppressLint("MissingPermission")
    override fun onResume() {
        super.onResume()
        // Start background thread for camera operations
        startBackgroundThread()
        if (textureView.isAvailable) {
            if (allPermissionsGranted()) {
                openCamera()
                updateShutterSpeed()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        } else {
            textureView.surfaceTextureListener = textureListener
        }
    }

    override fun onPause() {
        // If tracking active, stop it when pausing
        if (isTracking) {
            stopTracking()
        }
        closeCamera()
        stopBackgroundThread()
        super.onPause()
    }

    private fun stopTracking() {
        isTracking = false
        viewBinding.startTrackingButton.text = "Start Tracking"
        viewBinding.startTrackingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)

        videoProcessor.clearTrackingData()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    @RequiresPermission(Manifest.permission.CAMERA)
    private fun openCamera() {
        try {
            cameraId = getCameraId()
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            sensorArraySize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return
            previewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture::class.java))
            videoSize = chooseOptimalSize(map.getOutputSizes(MediaRecorder::class.java))
            cameraManager.openCamera(cameraId, stateCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        } catch (e: SecurityException) {
            e.printStackTrace()
            Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun chooseOptimalSize(choices: Array<Size>): Size {
        // For simplicity, pick the first available size
        return choices[0]
    }

    private fun getCameraId(): String {
        for (id in cameraManager.cameraIdList) {
            val characteristics = cameraManager.getCameraCharacteristics(id)
            val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
            if (facing == CameraCharacteristics.LENS_FACING_BACK && !isFrontCamera) {
                return id
            } else if (facing == CameraCharacteristics.LENS_FACING_FRONT && isFrontCamera) {
                return id
            }
        }
        return cameraManager.cameraIdList[0]
    }

    private fun switchCamera() {
        // If tracking active, stop first
        if (isTracking) stopTracking()
        isFrontCamera = !isFrontCamera
        closeCamera()
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)
        reopenCamera()
    }

    @SuppressLint("MissingPermission")
    private fun reopenCamera() {
        if (textureView.isAvailable) {
            if (allPermissionsGranted()) {
                openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        } else {
            textureView.surfaceTextureListener = textureListener
        }
    }

    private fun createCameraPreview() {
        try {
            val texture = textureView.surfaceTexture!!
            texture.setDefaultBufferSize(previewSize!!.width, previewSize!!.height)
            val surface = Surface(texture)
            captureRequestBuilder = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            captureRequestBuilder!!.addTarget(surface)

            // Apply various camera settings
            applyRollingShutter()
            updateShutterSpeed()
            applyFlashIfEnabled()
            applyLightingMode()
            applyZoom()

            cameraDevice!!.createCaptureSession(
                listOf(surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        if (cameraDevice == null) return
                        cameraCaptureSessions = session
                        updatePreview()
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Toast.makeText(this@MainActivity, "Configuration change", Toast.LENGTH_SHORT).show()
                    }
                },
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    private fun updatePreview() {
        if (cameraDevice == null) return
        try {
            captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            applyRollingShutter()
            updateShutterSpeed()
            applyFlashIfEnabled()
            applyLightingMode()
            cameraCaptureSessions?.setRepeatingRequest(captureRequestBuilder!!.build(), null, backgroundHandler)
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("Camera Background").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            e.printStackTrace()
        }
    }

    private fun closeCamera() {
        cameraCaptureSessions?.close()
        cameraCaptureSessions = null
        cameraDevice?.close()
        cameraDevice = null
    }

    private fun applyRollingShutter() {
        val shutterSpeedSetting = sharedPreferences.getString("shutter_speed", "15")?.toInt() ?: 15
        val shutterSpeedValue = if (shutterSpeedSetting >= 5) 1000000000L / shutterSpeedSetting else 0L
        captureRequestBuilder?.apply {
            if (shutterSpeedValue > 0) {
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterSpeedValue)
                Toast.makeText(this@MainActivity, "Rolling shutter speed applied: $shutterSpeedSetting Hz", Toast.LENGTH_SHORT).show()
            } else {
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                Toast.makeText(this@MainActivity, "Rolling shutter off", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        viewBinding.viewFinder.bitmap?.let { bitmap ->
            isProcessingFrame = true
            lifecycleScope.launch(Dispatchers.Default) {
                try {
                    // Process the frame in the background thread; lines drawn by OpenCV inside VideoProcessor
                    val processedBitmap = videoProcessor.processFrame(bitmap)
                    processedBitmap?.let { processedFrame ->
                        withContext(Dispatchers.Main) {
                            viewBinding.processedFrameView.setImageBitmap(processedFrame)
                        }

                        // TODO <Zaynab Mourtada>: Integrate PyTorch inference here if needed
                        // val inputTensor = bitmapToTensor(processedFrame)
                        // val outputTensor = runInference(inputTensor)
                        // processOutput(outputTensor, processedFrame.width, processedFrame.height)
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                } finally {
                    isProcessingFrame = false
                }
            }
        }
    }

    // TODO <Zaynab Mourtada>: Implement runInference logic with PyTorch if needed
    // private fun runInference(inputTensor: Tensor): Tensor {
    //     if (pytorchModule == null) {
    //         // TODO <Zaynab Mourtada>: Load PyTorch model first
    //     }
    //     // TODO <Zaynab Mourtada>: Replace with actual inference code
    //     return inputTensor
    // }

    // TODO <Zaynab Mourtada>: Implement bitmapToTensor logic for PyTorch model input
    // private fun bitmapToTensor(bitmap: Bitmap): Tensor {
    //     // TODO <Zaynab Mourtada>: Implement image preprocessing to tensor
    //     return ...
    // }

    // TODO <Zaynab Mourtada>: Implement processOutput to handle PyTorch inference results
    // private fun processOutput(outputTensor: Tensor, width: Int, height: Int): Bitmap {
    //     // TODO <Zaynab Mourtada>: Process the output of the model and overlay results
    //     return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    // }

    private val INPUT_WIDTH = 224
    private val INPUT_HEIGHT = 224
    // TODO <Zaynab Mourtada>: Adjust normalization if needed
    private val NO_MEAN_RGB = floatArrayOf(0.0f, 0.0f, 0.0f)
    private val NO_STD_RGB = floatArrayOf(1.0f, 1.0f, 1.0f)

    private fun updateShutterSpeed() {
        val shutterSpeedSetting = sharedPreferences.getString("shutter_speed", "15")?.toInt() ?: 15
        val shutterSpeedValue = 1000000000L / shutterSpeedSetting

        captureRequestBuilder?.apply {
            set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
            if (shutterSpeedSetting in 5..6000) {
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterSpeedValue)
            } else {
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            }

            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val exposureRange = characteristics.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)

            var exposureCompensation = 0
            if (exposureRange != null && shutterSpeedSetting in 5..6000) {
                val maxExposure = exposureRange.upper
                val minExposure = exposureRange.lower
                val normalizedShutterSpeed = (shutterSpeedSetting - 5).toFloat() / (6000 - 5)
                exposureCompensation = (minExposure + normalizedShutterSpeed * (maxExposure - minExposure)).toInt()
                set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, exposureCompensation)
            }

            try {
                cameraCaptureSessions?.setRepeatingRequest(build(), null, backgroundHandler)
                val shutterSpeedText = if (shutterSpeedSetting <= 6000) "1/$shutterSpeedSetting Hz" else "Auto"
                Log.d("RollingShutterUpdate","Set shutter speed to $shutterSpeedText with exposure compensation: $exposureCompensation")
                Toast.makeText(this@MainActivity,"Shutter speed set to $shutterSpeedText with exposure compensation: $exposureCompensation",Toast.LENGTH_SHORT).show()
            } catch (e: CameraAccessException) {
                e.printStackTrace()
            }
        }
    }

    private fun applyFlashIfEnabled() {
        val isFlashEnabled = sharedPreferences.getBoolean("enable_flash", false)
        captureRequestBuilder?.set(
            CaptureRequest.FLASH_MODE,
            if (isFlashEnabled) CaptureRequest.FLASH_MODE_TORCH else CaptureRequest.FLASH_MODE_OFF
        )
    }

    private fun applyLightingMode() {
        val lightingMode = sharedPreferences.getString("lighting_mode", "normal")
        val compensationRange = cameraManager.getCameraCharacteristics(cameraId)
            .get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)
        val exposureCompensation = when (lightingMode) {
            "low_light" -> compensationRange?.lower ?: 0
            "high_light" -> compensationRange?.upper ?: 0
            "normal" -> 0
            else -> 0
        }
        captureRequestBuilder?.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, exposureCompensation)
    }

    private fun setupZoomControls() {
        val zoomHandler = Handler(Looper.getMainLooper())
        var zoomInRunnable: Runnable? = null
        var zoomOutRunnable: Runnable? = null

        viewBinding.zoomInButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomInRunnable = object : Runnable {
                        override fun run() {
                            zoomIn()
                            zoomHandler.postDelayed(this, 50)
                        }
                    }
                    zoomHandler.post(zoomInRunnable!!)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    zoomHandler.removeCallbacks(zoomInRunnable!!)
                    true
                }
                else -> false
            }
        }

        viewBinding.zoomOutButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomOutRunnable = object : Runnable {
                        override fun run() {
                            zoomOut()
                            zoomHandler.postDelayed(this, 50)
                        }
                    }
                    zoomHandler.post(zoomOutRunnable!!)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    zoomHandler.removeCallbacks(zoomOutRunnable!!)
                    true
                }
                else -> false
            }
        }
    }

    private fun zoomIn() {
        if (zoomLevel < maxZoom) {
            zoomLevel += 0.1f
            applyZoom()
        }
    }

    private fun zoomOut() {
        if (zoomLevel > 1.0f) {
            zoomLevel -= 0.1f
            applyZoom()
        }
    }

    private fun applyZoom() {
        if (sensorArraySize == null || captureRequestBuilder == null) return
        val ratio = 1 / zoomLevel
        val croppedWidth = sensorArraySize!!.width() * ratio
        val croppedHeight = sensorArraySize!!.height() * ratio
        val zoomRect = Rect(
            ((sensorArraySize!!.width() - croppedWidth) / 2).toInt(),
            ((sensorArraySize!!.height() - croppedHeight) / 2).toInt(),
            ((sensorArraySize!!.width() + croppedWidth) / 2).toInt(),
            ((sensorArraySize!!.height() + croppedHeight) / 2).toInt()
        )
        captureRequestBuilder!!.set(CaptureRequest.SCALER_CROP_REGION, zoomRect)
        cameraCaptureSessions?.setRepeatingRequest(captureRequestBuilder!!.build(), null, backgroundHandler)
    }

    companion object {
        private const val SETTINGS_REQUEST_CODE = 1
        private val ORIENTATIONS = SparseIntArray()

        init {
            // Orientation map for camera sensor vs device rotation
            ORIENTATIONS.append(Surface.ROTATION_0, 90)
            ORIENTATIONS.append(Surface.ROTATION_90, 0)
            ORIENTATIONS.append(Surface.ROTATION_180, 270)
            ORIENTATIONS.append(Surface.ROTATION_270, 180)
        }
    }
}