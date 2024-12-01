package com.developer27.xamera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
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
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException


class MainActivity : AppCompatActivity() {

    // View binding for accessing UI components
    private lateinit var viewBinding: ActivityMainBinding

    // SharedPreferences for storing user settings
    private lateinit var sharedPreferences: SharedPreferences

    // Camera2 API variables
    private lateinit var cameraManager: CameraManager
    private var cameraDevice: CameraDevice? = null
    private lateinit var cameraId: String
    private var previewSize: Size? = null
    private var videoSize: Size? = null
    private var cameraCaptureSessions: CameraCaptureSession? = null
    private var captureRequestBuilder: CaptureRequest.Builder? = null

    // Flag to track whether tracking is active
    private var isTracking = false

    // Handler and thread for camera operations
    private var backgroundHandler: Handler? = null
    private var backgroundThread: HandlerThread? = null

    // Zoom variables
    private var zoomLevel = 1.0f
    private val maxZoom = 10.0f
    private var sensorArraySize: Rect? = null

    // Permissions required at startup
    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )

    // ActivityResultLauncher for permissions
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    // TextureView for camera preview
    private lateinit var textureView: TextureView

    // VideoProcessor instance for frame processing (assuming you have this class)
    private lateinit var videoProcessor: VideoProcessor

    // Flag to track which camera is currently active
    private var isFrontCamera = false

    // Shutter speed variable (default to 1/60s in nanoseconds)
    private var shutterSpeed: Long = 1000000000L / 60

    // Flag to prevent overlapping frame processing
    private var isProcessingFrame = false

    // OpenGLTextureView for OpenGL rendering
    private lateinit var glTextureView: OpenGLTextureView
    private var isOpenGLDemoActive = false

    // TODO <Zaynab Mourtada>: Declare your PyTorch Module
    private var pytorchModule: Module? = null

    // Listener for TextureView surface texture availability
    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            if (allPermissionsGranted()) {
                openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        }

        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
            // Handle size changes if needed
        }

        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
            // Return true if no further drawing should happen
            return false
        }

        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            // Process frame when tracking is active
            if (isTracking) {
                processFrameWithVideoProcessor()
            }
        }
    }

    // CameraDevice state callback
    private val stateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            // Camera is open, start preview
            cameraDevice = camera
            createCameraPreview()
        }

        override fun onDisconnected(camera: CameraDevice) {
            // Close camera when disconnected
            cameraDevice?.close()
            cameraDevice = null
        }

        override fun onError(camera: CameraDevice, error: Int) {
            // Close camera on error
            cameraDevice?.close()
            cameraDevice = null
            finish()
        }
    }

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Inflate layout and bind views
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Initialize the VideoProcessor
        videoProcessor = VideoProcessor(this)

        // Hide the processedFrameView initially
        viewBinding.processedFrameView.visibility = View.GONE

        // Initialize SharedPreferences
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)

        // Listen for changes in SharedPreferences
        sharedPreferences.registerOnSharedPreferenceChangeListener { prefs, key ->
            if (key == "shutter_speed") {
                // Get the updated shutter speed setting
                val shutterSpeedSetting = prefs.getString("shutter_speed", "60")?.toInt() ?: 60
                shutterSpeed = 1000000000L / shutterSpeedSetting // Convert to nanoseconds
                updateShutterSpeed() // Apply the new shutter speed
            }
        }

        // Load initial shutter speed from SharedPreferences
        val shutterSpeedSetting = sharedPreferences.getString("shutter_speed", "60")?.toInt() ?: 60
        shutterSpeed = 1000000000L / shutterSpeedSetting // Convert to nanoseconds

        // Initialize CameraManager
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

        // Initialize TextureView for camera preview
        textureView = viewBinding.viewFinder

        // Initialize OpenGLTextureView
        glTextureView = viewBinding.glTextureView

        // Initialize the ActivityResultLauncher for requesting permissions
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

        // Check permissions and set up camera preview
        if (allPermissionsGranted()) {
            if (textureView.isAvailable) {
                openCamera()
            } else {
                textureView.surfaceTextureListener = textureListener
            }
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        // Set up listener for the Start Tracking button
        viewBinding.startTrackingButton.setOnClickListener {
            toggleOpenGLDemo()
            if (isTracking) {
                stopTracking()
            } else {
                isTracking = true
                viewBinding.startTrackingButton.text = "Stop Tracking"
                viewBinding.startTrackingButton.backgroundTintList = ColorStateList.valueOf(
                    ContextCompat.getColor(this, R.color.red) // Change to red color
                )
                viewBinding.processedFrameView.visibility = View.VISIBLE
                videoProcessor.clearTrackingData()
            }
        }

        // Set up listener for the "Switch Camera" button
        viewBinding.switchCameraButton.setOnClickListener {
            switchCamera()
        }

        // Set up zoom controls
        setupZoomControls()

        // Set up listener for about button
        viewBinding.aboutButton.setOnClickListener {
            val intent = Intent(this, AboutXameraActivity::class.java)
            startActivity(intent)
        }

        // Set up settings button to open the settings activity
        viewBinding.settingsButton.setOnClickListener {
            val intent = Intent(this, SettingsActivity::class.java)
            startActivityForResult(intent, SETTINGS_REQUEST_CODE)
        }

        // TODO <Zaynab Mourtada>: Load your PyTorch model
        loadPyTorchModel()
    }

    // TODO <Zaynab Mourtada>: Function to load PyTorch Model
    private fun loadPyTorchModel() {
        // TODO <Zaynab Mourtada>: Replace "model.pt" with the actual filename of your model
        try {
            pytorchModule = Module.load(assetFilePath(this, "model.pt"))
        } catch (e: IOException) {
            Toast.makeText(this, "Error reading assets: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    // TODO <Zaynab Mourtada>: Function to get the asset file name
    @Throws(IOException::class)
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }

    // Handle results from activities (e.g., settings)
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == SETTINGS_REQUEST_CODE && resultCode == RESULT_OK) {
            // Reload the updated shutter speed
            val shutterSpeedSetting = sharedPreferences.getString("shutter_speed", "60")?.toInt() ?: 60
            shutterSpeed = 1000000000L / shutterSpeedSetting // Convert to nanoseconds
            Toast.makeText(this, "Shutter speed updated", Toast.LENGTH_SHORT).show()
            updateShutterSpeed()
        }
    }

    @SuppressLint("MissingPermission")
    override fun onResume() {
        super.onResume()
        startBackgroundThread()
        if (textureView.isAvailable) {
            if (allPermissionsGranted()) {
                openCamera()
                updateShutterSpeed()  // Apply the shutter speed when the camera opens
            } else {
                // Request permissions
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        } else {
            textureView.surfaceTextureListener = textureListener
        }
    }

    override fun onPause() {
        // Stop tracking if active
        if (isTracking) {
            stopTracking()
        }
        // Close camera and stop background thread
        closeCamera()
        stopBackgroundThread()
        super.onPause()
    }

    // Function to stop tracking
    private fun stopTracking() {
        isTracking = false
        viewBinding.startTrackingButton.text = "Start Tracking"
        viewBinding.startTrackingButton.backgroundTintList = ColorStateList.valueOf(
            ContextCompat.getColor(this, R.color.blue) // Revert to blue color
        )
        // Hide the processed camera view and clear any existing images
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)
        videoProcessor.clearTrackingData()
    }

    // Function to check if all required permissions are granted
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    // Function to open the camera
    @RequiresPermission(Manifest.permission.CAMERA)
    private fun openCamera() {
        try {
            cameraId = getCameraId()
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            sensorArraySize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            if (map == null) {
                Toast.makeText(this, "Cannot get available preview sizes", Toast.LENGTH_SHORT).show()
                return
            }
            previewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture::class.java))
            videoSize = chooseOptimalSize(map.getOutputSizes(MediaRecorder::class.java))

            // Use the Handler instead of Executor
            cameraManager.openCamera(cameraId, stateCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        } catch (e: SecurityException) {
            e.printStackTrace()
            Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_SHORT).show()
        }
    }

    // Function to choose the optimal size for preview
    private fun chooseOptimalSize(choices: Array<Size>): Size {
        // For simplicity, choose the first size available
        return choices[0]
    }

    // Function to get the camera ID based on the selected camera (front/back)
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
        return cameraManager.cameraIdList[0] // Default to first camera
    }

    // Function to switch between front and back camera
    private fun switchCamera() {
        // If tracking is active, stop it first
        if (isTracking) {
            stopTracking()
        }

        isFrontCamera = !isFrontCamera
        closeCamera()
        // Hide and clear the processed camera view when switching cameras
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)
        reopenCamera()
    }

    // Function to reopen the camera after switching
    @SuppressLint("MissingPermission")
    private fun reopenCamera() {
        if (textureView.isAvailable) {
            if (allPermissionsGranted()) {
                openCamera()
            } else {
                // Request permissions
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        } else {
            textureView.surfaceTextureListener = textureListener
        }
    }

    // Function to create camera preview
    private fun createCameraPreview() {
        try {
            val texture = textureView.surfaceTexture!!
            texture.setDefaultBufferSize(previewSize!!.width, previewSize!!.height)
            val surface = Surface(texture)
            captureRequestBuilder = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            captureRequestBuilder!!.addTarget(surface)

            // Apply necessary settings
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

    // Function to update camera preview
    private fun updatePreview() {
        if (cameraDevice == null) return
        try {
            captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)

            // Apply necessary settings
            applyRollingShutter()
            updateShutterSpeed()
            applyFlashIfEnabled()
            applyLightingMode()

            cameraCaptureSessions?.setRepeatingRequest(captureRequestBuilder!!.build(), null, backgroundHandler)
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    // Function to start background thread
    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("Camera Background").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    // Function to stop background thread
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

    // Function to close the camera
    private fun closeCamera() {
        cameraCaptureSessions?.close()
        cameraCaptureSessions = null
        cameraDevice?.close()
        cameraDevice = null
    }

    // Function to apply rolling shutter speed
    private fun applyRollingShutter() {
        // Retrieve the selected shutter speed setting from SharedPreferences
        val shutterSpeedSetting = sharedPreferences.getString("shutter_speed", "15")?.toInt() ?: 15
        val shutterSpeedValue = if (shutterSpeedSetting >= 5) 1000000000L / shutterSpeedSetting else 0L

        captureRequestBuilder?.apply {
            if (shutterSpeedValue > 0) {
                // Apply the calculated shutter speed based on user setting
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterSpeedValue)
                Toast.makeText(this@MainActivity, "Rolling shutter speed applied: $shutterSpeedSetting Hz", Toast.LENGTH_SHORT).show()
            } else {
                // If "Off" is selected, use default automatic mode
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                Toast.makeText(this@MainActivity, "Rolling shutter off", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // Function to process frames - Calls videoProcessor.processFrame(bitmap)
    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        viewBinding.viewFinder.bitmap?.let { bitmap ->
            isProcessingFrame = true
            lifecycleScope.launch(Dispatchers.Default) {
                try {
                    // First, process the bitmap with videoProcessor
                    val processedBitmap = videoProcessor.processFrame(bitmap)
                    processedBitmap?.let { processedFrame ->
                        withContext(Dispatchers.Main) {
                            // Update the UI with the processed bitmap from videoProcessor
                            viewBinding.processedFrameView.setImageBitmap(processedFrame)
                        }

                        // TODO <Zaynab Mourtada>: Now, run PyTorch inference on the processed frame
                        val inputTensor = bitmapToTensor(processedFrame)
                        val outputTensor = runInference(inputTensor)
                        val pytorchProcessedBitmap = processOutput(outputTensor, processedFrame.width, processedFrame.height)
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                } finally {
                    isProcessingFrame = false
                }
            }
        }
    }

    // TODO <Zaynab Mourtada>: Use the PyTorch model in your code
    private fun runInference(inputTensor: Tensor): Tensor {
        // Ensure the model is loaded
        if (pytorchModule == null) {
            loadPyTorchModel()
        }

        // Run the model inference
        val outputTensor = pytorchModule!!.forward(IValue.from(inputTensor)).toTensor()
        return outputTensor
    }

    // TODO <Zaynab Mourtada>: Function to transform bitmap to Tensor
    private fun bitmapToTensor(bitmap: Bitmap): Tensor {
        // Preprocess the bitmap as per your model's requirement
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_WIDTH, INPUT_HEIGHT, true)
        return TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            NO_MEAN_RGB,
            NO_STD_RGB
        )
    }

    // TODO <Zaynab Mourtada>: Constants for input size and normalization
    private val INPUT_WIDTH = 224  // Replace with your model's input width
    private val INPUT_HEIGHT = 224 // Replace with your model's input height
    private val NO_MEAN_RGB = floatArrayOf(0.0f, 0.0f, 0.0f)
    private val NO_STD_RGB = floatArrayOf(1.0f, 1.0f, 1.0f)

    // TODO <Zaynab Mourtada>: Algorithm to process the output
    private fun processOutput(outputTensor: Tensor, width: Int, height: Int): Bitmap {
        // TODO <Zaynab Mourtada>: Implement your processing logic here using outputTensor, width, and height

        // Create a new bitmap with the specified width and height
        val resultBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(resultBitmap)
        val paint = Paint()

        // Example processing based on the output tensor
        val scores = outputTensor.dataAsFloatArray

        // For demonstration, let's fill the bitmap with a color based on the first score
        if (scores.isNotEmpty()) {
            val colorValue = (scores[0] * 255).toInt().coerceIn(0, 255)
            val color = Color.rgb(colorValue, 0, 255 - colorValue)
            canvas.drawColor(color)
        } else {
            // Default color if scores are empty
            canvas.drawColor(Color.BLACK)
        }

        return resultBitmap
    }

    // Function to update shutter speed
    private fun updateShutterSpeed() {
        // Retrieve the selected shutter speed from SharedPreferences
        val shutterSpeedSetting = sharedPreferences.getString("shutter_speed", "15")?.toInt() ?: 15
        val shutterSpeedValue = 1000000000L / shutterSpeedSetting // Convert to nanoseconds

        captureRequestBuilder?.apply {
            set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF) // Manual control

            if (shutterSpeedSetting in 5..6000) { // Ensure the setting is valid
                // Apply the calculated shutter speed for selected Hz values
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterSpeedValue)
            } else {
                // Default to auto control if shutter speed is invalid (below 5 Hz)
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            }

            // Retrieve the camera's exposure compensation range
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val exposureRange = characteristics.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)

            // Apply exposure compensation based on selected shutter speed value
            var exposureCompensation = 0
            if (exposureRange != null && shutterSpeedSetting in 5..6000) {
                // Interpolate exposure compensation based on shutter speed
                val maxExposure = exposureRange.upper
                val minExposure = exposureRange.lower

                // Calculate compensation based on normalized shutter speed
                val normalizedShutterSpeed = (shutterSpeedSetting - 5).toFloat() / (6000 - 5) // Scale between 0.0 and 1.0
                exposureCompensation = (minExposure + normalizedShutterSpeed * (maxExposure - minExposure)).toInt()

                set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, exposureCompensation)
            }

            try {
                // Apply the capture request with updated settings
                cameraCaptureSessions?.setRepeatingRequest(build(), null, backgroundHandler)

                // Log and Toast messages to verify the correct values
                val shutterSpeedText = if (shutterSpeedSetting <= 6000) "1/$shutterSpeedSetting Hz" else "Auto"
                Log.d(
                    "RollingShutterUpdate",
                    "Set shutter speed to $shutterSpeedText with exposure compensation: $exposureCompensation"
                )
                Toast.makeText(
                    this@MainActivity,
                    "Shutter speed set to $shutterSpeedText with exposure compensation: $exposureCompensation",
                    Toast.LENGTH_SHORT
                ).show()
            } catch (e: CameraAccessException) {
                e.printStackTrace()
            }
        }
    }

    // Function to apply flash if enabled
    private fun applyFlashIfEnabled() {
        val isFlashEnabled = sharedPreferences.getBoolean("enable_flash", false)
        captureRequestBuilder?.set(
            CaptureRequest.FLASH_MODE,
            if (isFlashEnabled) CaptureRequest.FLASH_MODE_TORCH else CaptureRequest.FLASH_MODE_OFF
        )
    }

    // Function to apply lighting mode
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
        captureRequestBuilder?.set(
            CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION,
            exposureCompensation
        )
    }

    // Function to set up zoom controls with continuous zoom
    private fun setupZoomControls() {
        val zoomHandler = Handler(Looper.getMainLooper())
        var zoomInRunnable: Runnable? = null
        var zoomOutRunnable: Runnable? = null

        // Continuous zoom in on long press
        viewBinding.zoomInButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomInRunnable = object : Runnable {
                        override fun run() {
                            zoomIn()
                            zoomHandler.postDelayed(this, 50) // Adjust delay as needed
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

        // Continuous zoom out on long press
        viewBinding.zoomOutButton.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    zoomOutRunnable = object : Runnable {
                        override fun run() {
                            zoomOut()
                            zoomHandler.postDelayed(this, 50) // Adjust delay as needed
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

    // Function to zoom in
    private fun zoomIn() {
        if (zoomLevel < maxZoom) {
            zoomLevel += 0.1f
            applyZoom()
        }
    }

    // Function to zoom out
    private fun zoomOut() {
        if (zoomLevel > 1.0f) {
            zoomLevel -= 0.1f
            applyZoom()
        }
    }

    // Function to apply zoom
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

    // Companion object for constants
    companion object {
        private const val SETTINGS_REQUEST_CODE = 1

        // Tag for logging
        private const val TAG = "Xamera"

        // Orientation constants
        private val ORIENTATIONS = SparseIntArray()
        init {
            ORIENTATIONS.append(Surface.ROTATION_0, 90)
            ORIENTATIONS.append(Surface.ROTATION_90, 0)
            ORIENTATIONS.append(Surface.ROTATION_180, 270)
            ORIENTATIONS.append(Surface.ROTATION_270, 180)
        }
    }

    // Function to toggle the OpenGL Demo
    private fun toggleOpenGLDemo() {
        if (isOpenGLDemoActive) {
            glTextureView.visibility = View.GONE
            isOpenGLDemoActive = false
        } else {
            glTextureView.visibility = View.VISIBLE
            isOpenGLDemoActive = true
        }
    }
}
