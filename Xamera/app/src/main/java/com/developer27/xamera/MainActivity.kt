package com.developer27.xamera

// TODO <Zaynab Mourtada>: Uncomment these once you have PyTorch Mobile in your build.gradle
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
import android.widget.EditText
import android.widget.Switch
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresPermission
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.developer27.xamera.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

/**
 * MainActivity for the Xamera app.
 *
 * - Shows camera preview
 * - Handles tracking start/stop, camera switching, zoom, etc.
 * - Loads TWO .pt models at startup: handwriting_lstm_model.pt and digits_lstm_model.pt
 * - DOES NOT do “live” predictions while the user is drawing.
 *   Instead, it collects coordinates and does final predictions after user stops tracking.
 *
 * Now, we also ask the user to confirm if the prediction is correct. If not,
 * we let them enter the correct label and we store the data for future training.
 */
class MainActivity : AppCompatActivity() {

    // ------------------------------------------------------------------------------------
    // Fields and references
    // ------------------------------------------------------------------------------------
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager

    private var cameraDevice: CameraDevice? = null
    private lateinit var cameraId: String
    private var previewSize: Size? = null
    private var videoSize: Size? = null
    private var cameraCaptureSessions: CameraCaptureSession? = null
    private var captureRequestBuilder: CaptureRequest.Builder? = null

    // Tracking states
    private var isTracking = false
    private var isFrontCamera = false
    private var isProcessingFrame = false

    // Background thread
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null

    // Zoom
    private var zoomLevel = 1.0f
    private val maxZoom = 10.0f
    private var sensorArraySize: Rect? = null

    // Required permissions
    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    // The TextureView for camera preview
    private lateinit var textureView: TextureView

    // The video processor that collects (X,Y,Frame)
    private lateinit var videoProcessor: VideoProcessor

    // If using an OpenGLTextureView for advanced rendering
    private lateinit var glTextureView: OpenGLTextureView

    // Shutter speed
    private var shutterSpeed: Long = 1000000000L / 60

    // A reference to the Switch in our layout
    private lateinit var switchDisplayMode: Switch

    // ------------------------------------------------------------------------------------
    // (OPTIONAL) Two PyTorch models:
    //   1) handwritingModule for letters
    //   2) digitsModule for digits 0–9
    // Uncomment the lines if using PyTorch in build.gradle
    // ------------------------------------------------------------------------------------
//    private var handwritingModule: Module? = null
//    private var digitsModule: Module? = null

    private val letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    // ------------------------------------------------------------------------------------
    // TextureView listener
    // ------------------------------------------------------------------------------------
    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            if (allPermissionsGranted()) {
                openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean { return false }
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            // Called each new camera frame
            if (isTracking) {
                // Process frame for drawing, but no final inference
                processFrameWithVideoProcessor()
            }
        }
    }

    // Camera state callback
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

    // ------------------------------------------------------------------------------------
    // onCreate
    // ------------------------------------------------------------------------------------
    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Inflate binding
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Hide processed frame until tracking starts
        viewBinding.processedFrameView.visibility = View.GONE

        // SharedPreferences
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)

        // Listen for changes in certain preferences (like shutter_speed).
        sharedPreferences.registerOnSharedPreferenceChangeListener { prefs, key ->
            if (key == "shutter_speed") {
                val shutterSpeedSetting = prefs.getString("shutter_speed", "60")?.toInt() ?: 60
                shutterSpeed = 1000000000L / shutterSpeedSetting
                updateShutterSpeed()
            }
        }

        // Initialize shutter speed from preferences
        val shutterSpeedSetting = sharedPreferences.getString("shutter_speed", "60")?.toInt() ?: 60
        shutterSpeed = 1000000000L / shutterSpeedSetting

        // Initialize cameraManager
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

        // Initialize the views
        textureView = viewBinding.viewFinder
        videoProcessor = VideoProcessor(this)
        glTextureView = viewBinding.glTextureView

        // Find the Switch from the layout
        switchDisplayMode = findViewById(R.id.switchDisplayMode)
        // By default, if it's checked => show digits, otherwise => show letters

        // Permission launcher
        requestPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            val camGranted = permissions[Manifest.permission.CAMERA] ?: false
            val micGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
            if (camGranted && micGranted) {
                if (textureView.isAvailable) openCamera()
                else textureView.surfaceTextureListener = textureListener
            } else {
                Toast.makeText(
                    this,
                    "Camera and Audio permissions are required.",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }

        // Check permissions
        if (allPermissionsGranted()) {
            if (textureView.isAvailable) openCamera()
            else textureView.surfaceTextureListener = textureListener
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        // Buttons
        viewBinding.startTrackingButton.setOnClickListener {
            if (isTracking) stopTracking() else startTracking()
        }
        viewBinding.switchCameraButton.setOnClickListener {
            switchCamera()
        }
        setupZoomControls()
        viewBinding.aboutButton.setOnClickListener {
            val intent = Intent(this, AboutXameraActivity::class.java)
            startActivity(intent)
        }
        viewBinding.settingsButton.setOnClickListener {
            val intent = Intent(this, SettingsActivity::class.java)
            startActivityForResult(intent, SETTINGS_REQUEST_CODE)
        }

        // Load both models at startup (example placeholders):
        loadAllModelsOnStartup(
            handwritingModel = "handwriting_lstm_model.pt",
            digitsModel = "digits_lstm_model.pt"
        )
    }

    /**
     * Load BOTH models from the app’s assets, copy them to internal storage,
     * and optionally load them into PyTorch Modules (uncomment if using PyTorch).
     */
    private fun loadAllModelsOnStartup(handwritingModel: String, digitsModel: String) {
        lifecycleScope.launch(Dispatchers.IO) {
            val handwritingLoaded = copyAssetModel(handwritingModel)
            val digitsLoaded = copyAssetModel(digitsModel)

            // If using PyTorch, load them, e.g.:
            // if (handwritingLoaded.isNotEmpty()) {
            //     handwritingModule = Module.load(handwritingLoaded)
            // }
            // if (digitsLoaded.isNotEmpty()) {
            //     digitsModule = Module.load(digitsLoaded)
            // }

            withContext(Dispatchers.Main) {
                if (handwritingLoaded.isEmpty() && digitsLoaded.isEmpty()) {
                    Toast.makeText(
                        this@MainActivity,
                        "Failed to load BOTH handwriting & digits models.",
                        Toast.LENGTH_LONG
                    ).show()
                } else {
                    Toast.makeText(
                        this@MainActivity,
                        "Models loaded (handwriting & digits).",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
    }

    /**
     * Copy a given model file from assets to internal storage.
     * Returns the absolute path if successful, or "" if it fails.
     */
    private suspend fun copyAssetModel(assetName: String): String {
        return withContext(Dispatchers.IO) {
            try {
                val outFile = File(filesDir, assetName)
                if (outFile.exists() && outFile.length() > 0) {
                    return@withContext outFile.absolutePath
                }
                assets.open(assetName).use { input ->
                    FileOutputStream(outFile).use { output ->
                        val buffer = ByteArray(4 * 1024)
                        var bytesRead: Int
                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            output.write(buffer, 0, bytesRead)
                        }
                        output.flush()
                    }
                }
                outFile.absolutePath
            } catch (e: Exception) {
                Log.e("MainActivity", "Error copying asset $assetName: ${e.message}")
                ""
            }
        }
    }

    // ------------------------------------------------------------------------------------
    // This is called each frame if isTracking == true. We do NOT do final inference here.
    // ------------------------------------------------------------------------------------
    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true

        lifecycleScope.launch(Dispatchers.Default) {
            val processedBitmap = videoProcessor.processFrame(bitmap)
            if (processedBitmap != null) {
                withContext(Dispatchers.Main) {
                    viewBinding.processedFrameView.setImageBitmap(processedBitmap)
                }
            }
            isProcessingFrame = false
        }
    }

    // ------------------------------------------------------------------------------------
    // Start/Stop Tracking
    // ------------------------------------------------------------------------------------
    private fun startTracking() {
        isTracking = true
        viewBinding.startTrackingButton.text = "Stop Tracking"
        viewBinding.startTrackingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)

        viewBinding.processedFrameView.visibility = View.VISIBLE
        videoProcessor.clearTrackingData()

        Toast.makeText(this, "Tracking started. Data is being collected.", Toast.LENGTH_SHORT).show()
    }

    /**
     * We parse the finalPrediction here, then decide if we display
     * the letter part or digit part based on our Switch (switchDisplayMode).
     * Then we ask "Is this correct?" If No => let user fix the label.
     */
    private fun stopTracking() {
        isTracking = false
        viewBinding.startTrackingButton.text = "Start Tracking"
        viewBinding.startTrackingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)

        // Hide or clear the processed frame
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)

        Toast.makeText(this, "Tracking stopped. Processing data...", Toast.LENGTH_SHORT).show()

        lifecycleScope.launch(Dispatchers.Main) {
            kotlinx.coroutines.delay(500) // short delay
            val finalPrediction = withContext(Dispatchers.Default) {
                videoProcessor.predictFromCollectedData()
            }
            if (finalPrediction != null) {
                // Parse out "Letter: X" vs. "Digit: Y"
                val lines = finalPrediction.split("\n")
                var letterResult = ""
                var digitResult = ""
                for (line in lines) {
                    if (line.startsWith("Letter: ")) {
                        letterResult = line.substringAfter("Letter: ").trim()
                    } else if (line.startsWith("Digit: ")) {
                        digitResult = line.substringAfter("Digit: ").trim()
                    }
                }

                val wantsDigits = switchDisplayMode.isChecked
                val predictedString = if (wantsDigits) {
                    "Digit: $digitResult"
                } else {
                    "Letter: $letterResult"
                }

                // Show a dialog => "Is this correct? Yes / No"
                AlertDialog.Builder(this@MainActivity)
                    .setTitle("Final Prediction")
                    .setMessage("We guessed: $predictedString\nIs that correct?")
                    .setPositiveButton("Yes") { dialog, _ ->
                        // If yes => do nothing special
                        dialog.dismiss()
                    }
                    .setNegativeButton("No") { dialog, _ ->
                        dialog.dismiss()
                        // If no => ask user for correction
                        askUserForCorrection(letterResult, digitResult, wantsDigits)
                    }
                    .create()
                    .show()
            } else {
                val dialog = AlertDialog.Builder(this@MainActivity)
                    .setTitle("Prediction")
                    .setMessage("Cannot predict from tracked data.")
                    .setPositiveButton("OK") { d, _ -> d.dismiss() }
                    .create()
                dialog.show()
            }
        }
    }

    // ------------------------------------------------------------------------------------
    // If user says "No," ask for the correct label (via an EditText),
    // then store all postFilter4Ddata + that label to a local CSV for training.
    // ------------------------------------------------------------------------------------
    private fun askUserForCorrection(
        guessedLetter: String,
        guessedDigit: String,
        wasDigitMode: Boolean
    ) {
        val editText = EditText(this)
        editText.hint = if (wasDigitMode) "Enter correct digit (0–9)" else "Enter correct letter (A–Z)"

        AlertDialog.Builder(this)
            .setTitle("Please Enter the Correct Label")
            .setView(editText)
            .setPositiveButton("OK") { dialog, _ ->
                val userInput = editText.text.toString().trim()
                dialog.dismiss()

                // 1) Save all (X, Y, Frame) from videoProcessor’s postFilter4Ddata
                // plus the userInput as the correct label.
                lifecycleScope.launch(Dispatchers.IO) {
                    saveCorrectionToCSV(userInput, wasDigitMode)
                }

                // 2) Toast
                Toast.makeText(
                    this,
                    "Saved correction: $userInput.",
                    Toast.LENGTH_SHORT
                ).show()
            }
            .setNegativeButton("Cancel") { d, _ -> d.dismiss() }
            .create()
            .show()
    }

    // ------------------------------------------------------------------------------------
    // saveCorrectionToCSV
    // Writes all postFilter4Ddata to a file "corrections.csv" with columns:
    // X, Y, Frame, isDigit, userLabel
    // ------------------------------------------------------------------------------------
    private suspend fun saveCorrectionToCSV(
        userLabel: String,
        isDigit: Boolean
    ) = withContext(Dispatchers.IO) {
        try {
            // 1) Build a path for storing corrections.
            val correctionsFile = File(filesDir, "corrections.csv")
            val isNewFile = !correctionsFile.exists()

            // 2) Gather data from videoProcessor
            val allData = videoProcessor.getPostFilterData()

            // 3) Append text
            correctionsFile.appendText(buildString {
                // If new => header
                if (isNewFile) {
                    appendLine("X,Y,Frame,IsDigit,UserLabel")
                }
                for (fd in allData) {
                    appendLine("${fd.x},${fd.y},${fd.frameCount},$isDigit,$userLabel")
                }
            })
            Log.i("MainActivity", "Appended correction for label=$userLabel to ${correctionsFile.absolutePath}")
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to save correction CSV: ${e.message}")
        }
    }

    // ------------------------------------------------------------------------------------
    // onActivityResult
    // ------------------------------------------------------------------------------------
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == SETTINGS_REQUEST_CODE && resultCode == RESULT_OK) {
            // Possibly the shutter speed changed in Settings
            val shutterSpeedSetting = sharedPreferences.getString("shutter_speed", "60")?.toInt() ?: 60
            shutterSpeed = 1000000000L / shutterSpeedSetting
            Toast.makeText(this, "Settings updated", Toast.LENGTH_SHORT).show()
            updateShutterSpeed()
        }
    }

    // ------------------------------------------------------------------------------------
    // onResume / onPause
    // ------------------------------------------------------------------------------------
    @SuppressLint("MissingPermission")
    override fun onResume() {
        super.onResume()
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
        if (isTracking) stopTracking()
        closeCamera()
        stopBackgroundThread()
        super.onPause()
    }

    // ------------------------------------------------------------------------------------
    // Check permissions
    // ------------------------------------------------------------------------------------
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    // ------------------------------------------------------------------------------------
    // Camera logic
    // ------------------------------------------------------------------------------------
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
            val facing = cameraManager.getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING)
            if (facing == CameraCharacteristics.LENS_FACING_BACK && !isFrontCamera) {
                return id
            } else if (facing == CameraCharacteristics.LENS_FACING_FRONT && isFrontCamera) {
                return id
            }
        }
        // fallback
        return cameraManager.cameraIdList[0]
    }

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

    private fun closeCamera() {
        cameraCaptureSessions?.close()
        cameraCaptureSessions = null
        cameraDevice?.close()
        cameraDevice = null
    }

    private fun switchCamera() {
        if (isTracking) stopTracking()
        isFrontCamera = !isFrontCamera
        closeCamera()
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)
        reopenCamera()
    }

    private fun createCameraPreview() {
        try {
            val texture = textureView.surfaceTexture ?: return
            texture.setDefaultBufferSize(previewSize!!.width, previewSize!!.height)
            val surface = Surface(texture)

            captureRequestBuilder = cameraDevice!!.createCaptureRequest(
                CameraDevice.TEMPLATE_PREVIEW
            ).apply { addTarget(surface) }

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
                        Toast.makeText(
                            this@MainActivity,
                            "Configuration change",
                            Toast.LENGTH_SHORT
                        ).show()
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
            captureRequestBuilder?.set(
                CaptureRequest.CONTROL_MODE,
                CameraMetadata.CONTROL_MODE_AUTO
            )
            applyRollingShutter()
            updateShutterSpeed()
            applyFlashIfEnabled()
            applyLightingMode()

            cameraCaptureSessions?.setRepeatingRequest(
                captureRequestBuilder!!.build(),
                null,
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    // ------------------------------------------------------------------------------------
    // Background thread
    // ------------------------------------------------------------------------------------
    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackgroundThread").also { it.start() }
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

    // ------------------------------------------------------------------------------------
    // Rolling shutter, flash, lighting, zoom
    // ------------------------------------------------------------------------------------
    private fun applyRollingShutter() {
        val shutterSetting = sharedPreferences.getString("shutter_speed", "15")?.toInt() ?: 15
        val shutterValue = if (shutterSetting >= 5) 1000000000L / shutterSetting else 0L

        captureRequestBuilder?.apply {
            if (shutterValue > 0) {
                // Turn OFF auto mode so we can set manual exposure
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterValue)

                // A friendlier Toast explaining rolling shutter:
                Toast.makeText(
                    this@MainActivity,
                    "Rolling shutter ON at about 1/$shutterSetting second per exposure.\n" +
                            "This reduces motion blur but may result in darker images.",
                    Toast.LENGTH_LONG
                ).show()
            } else {
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                // Friendlier Toast for OFF
                Toast.makeText(
                    this@MainActivity,
                    "Rolling shutter is OFF (auto exposure).",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun applyFlashIfEnabled() {
        val isFlashEnabled = sharedPreferences.getBoolean("enable_flash", false)
        captureRequestBuilder?.set(
            CaptureRequest.FLASH_MODE,
            if (isFlashEnabled) CaptureRequest.FLASH_MODE_TORCH
            else CaptureRequest.FLASH_MODE_OFF
        )
    }

    private fun applyLightingMode() {
        val lightingMode = sharedPreferences.getString("lighting_mode", "normal")
        val compensationRange =
            cameraManager.getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)

        val exposureComp = when (lightingMode) {
            "low_light" -> compensationRange?.lower ?: 0
            "high_light" -> compensationRange?.upper ?: 0
            else -> 0
        }
        captureRequestBuilder?.set(
            CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION,
            exposureComp
        )
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
        cameraCaptureSessions?.setRepeatingRequest(
            captureRequestBuilder!!.build(),
            null,
            backgroundHandler
        )
    }

    /**
     * Update the shutter speed after user changes it in Settings.
     *
     * We also try to give the user a short Toast about the shutter speed,
     * but the main explanation is in applyRollingShutter().
     */
    private fun updateShutterSpeed() {
        val shutterSetting = sharedPreferences.getString("shutter_speed", "15")?.toInt() ?: 15
        val shutterValue = 1000000000L / shutterSetting

        captureRequestBuilder?.apply {
            set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
            if (shutterSetting in 5..6000) {
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterValue)
            } else {
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            }

            // Still call repeatingRequest so it updates if needed
            try {
                cameraCaptureSessions?.setRepeatingRequest(build(), null, backgroundHandler)
                // A small toast to mention the new setting
                if (shutterSetting <= 6000) {
                    Toast.makeText(
                        this@MainActivity,
                        "Shutter speed updated to about 1/$shutterSetting second per exposure.",
                        Toast.LENGTH_SHORT
                    ).show()
                } else {
                    Toast.makeText(
                        this@MainActivity,
                        "Shutter speed set to AUTO.",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            } catch (e: CameraAccessException) {
                e.printStackTrace()
            }
        }
    }

    companion object {
        private const val SETTINGS_REQUEST_CODE = 1
        private val ORIENTATIONS = SparseIntArray()
        init {
            ORIENTATIONS.append(Surface.ROTATION_0, 90)
            ORIENTATIONS.append(Surface.ROTATION_90, 0)
            ORIENTATIONS.append(Surface.ROTATION_180, 270)
            ORIENTATIONS.append(Surface.ROTATION_270, 180)
        }
    }
}
