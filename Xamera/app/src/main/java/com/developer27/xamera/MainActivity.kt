package com.developer27.xamera

// TODO <Zaynab Mourtada>: Uncomment these once you have PyTorch Mobile in your build.gradle
// import org.pytorch.IValue
// import org.pytorch.Module
// import org.pytorch.Tensor
// import org.pytorch.torchvision.TensorImageUtils

// If you need coroutines or advanced logic:
// import androidx.lifecycle.lifecycleScope
// import kotlinx.coroutines.Dispatchers
// import kotlinx.coroutines.launch
// import kotlinx.coroutines.withContext
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
import android.os.Environment
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
import android.widget.Switch
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresPermission
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.developer27.xamera.databinding.ActivityMainBinding
import java.io.File

/**
 * MainActivity for the Xamera app:
 * - Uses rolling-shutter for both preview and recording
 * - Saves video to public Movies folder
 * - Has references to PyTorch & VideoProcessor code (commented out)
 */
class MainActivity : AppCompatActivity() {

    // ------------------------------------------------------------------------------------
    // Fields
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

    // Rolling-shutter + background thread
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

    // TextureView for preview
    private lateinit var textureView: TextureView

    // Shutter speed
    private var shutterSpeed: Long = 1000000000L / 60

    // A Switch in the layout
    private lateinit var switchDisplayMode: Switch

    // MediaRecorder
    private var mediaRecorder: MediaRecorder? = null
    private var isRecordingVideo = false
    private var outputFile: File? = null

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

    // ------------------------------------------------------------------------------------
    // (OPTIONAL) Pytorch references from older code (commented out)
    // ------------------------------------------------------------------------------------
//    private var handwritingModule: Module? = null
//    private var digitsModule: Module? = null
//    private val letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

//    private fun loadAllModelsOnStartup(handwritingModel: String, digitsModel: String) {
//        lifecycleScope.launch(Dispatchers.IO) {
//            val handwritingLoaded = copyAssetModel(handwritingModel)
//            val digitsLoaded = copyAssetModel(digitsModel)
//
//            // if (handwritingLoaded.isNotEmpty()) {
//            //     handwritingModule = Module.load(handwritingLoaded)
//            // }
//            // if (digitsLoaded.isNotEmpty()) {
//            //     digitsModule = Module.load(digitsLoaded)
//            // }
//
//            withContext(Dispatchers.Main) {
//                Toast.makeText(
//                    this@MainActivity,
//                    "Models loaded (handwriting & digits).",
//                    Toast.LENGTH_SHORT
//                ).show()
//            }
//        }
//    }
//
//    private suspend fun copyAssetModel(assetName: String): String {
//        return withContext(Dispatchers.IO) {
//            try {
//                val outFile = File(filesDir, assetName)
//                if (outFile.exists() && outFile.length() > 0) {
//                    return@withContext outFile.absolutePath
//                }
//                assets.open(assetName).use { input ->
//                    FileOutputStream(outFile).use { output ->
//                        val buffer = ByteArray(4 * 1024)
//                        var bytesRead: Int
//                        while (input.read(buffer).also { bytesRead = it } != -1) {
//                            output.write(buffer, 0, bytesRead)
//                        }
//                        output.flush()
//                    }
//                }
//                outFile.absolutePath
//            } catch (e: Exception) {
//                Log.e("MainActivity", "Error copying asset $assetName: ${e.message}")
//                ""
//            }
//        }
//    }

    // ------------------------------------------------------------------------------------
    // VideoProcessor references from older code (commented out)
    // ------------------------------------------------------------------------------------
//    private var isProcessingFrame = false
//    private lateinit var videoProcessor: VideoProcessor
//
//    private fun processFrameWithVideoProcessor() {
//        if (isProcessingFrame) return
//        val bitmap = viewBinding.viewFinder.bitmap ?: return
//        isProcessingFrame = true
//
//        lifecycleScope.launch(Dispatchers.Default) {
//            val processedBitmap = videoProcessor.processFrame(bitmap)
//            if (processedBitmap != null) {
//                withContext(Dispatchers.Main) {
//                    viewBinding.processedFrameView.setImageBitmap(processedBitmap)
//                }
//            }
//            isProcessingFrame = false
//        }
//    }
//
//    /**
//     * If user says "No," ask for the correct label (via an EditText),
//     * then store all postFilter4Ddata + that label to a local CSV for training.
//     */
//    private fun askUserForCorrection(
//        guessedLetter: String,
//        guessedDigit: String,
//        wasDigitMode: Boolean
//    ) {
//        /*
//        val editText = EditText(this)
//        editText.hint = if (wasDigitMode) "Enter correct digit (0–9)" else "Enter correct letter (A–Z)"
//
//        AlertDialog.Builder(this)
//            .setTitle("Please Enter the Correct Label")
//            .setView(editText)
//            .setPositiveButton("OK") { dialog, _ ->
//                val userInput = editText.text.toString().trim()
//                dialog.dismiss()
//
//                // 1) Save all (X, Y, Frame) from videoProcessor’s postFilter4Ddata
//                //    plus the userInput as the correct label.
//                lifecycleScope.launch(Dispatchers.IO) {
//                    saveCorrectionToCSV(userInput, wasDigitMode)
//                }
//
//                // 2) Toast
//                Toast.makeText(
//                    this,
//                    "Saved correction: $userInput.",
//                    Toast.LENGTH_SHORT
//                ).show()
//            }
//            .setNegativeButton("Cancel") { d, _ -> d.dismiss() }
//            .create()
//            .show()
//         */
//    }
//
//    /**
//     * Writes all postFilter4Ddata to "corrections.csv" with columns: X, Y, Frame, isDigit, userLabel.
//     */
//    private suspend fun saveCorrectionToCSV(
//        userLabel: String,
//        isDigit: Boolean
//    ) = withContext(Dispatchers.IO) {
//        /*
//        try {
//            // 1) Build a path for storing corrections.
//            val correctionsFile = File(filesDir, "corrections.csv")
//            val isNewFile = !correctionsFile.exists()
//
//            // 2) Gather data from videoProcessor
//            val allData = videoProcessor.getPostFilterData()
//
//            // 3) Append text
//            correctionsFile.appendText(buildString {
//                // If new => header
//                if (isNewFile) {
//                    appendLine("X,Y,Frame,IsDigit,UserLabel")
//                }
//                for (fd in allData) {
//                    appendLine("${fd.x},${fd.y},${fd.frameCount},$isDigit,$userLabel")
//                }
//            })
//            Log.i("MainActivity", "Appended correction for label=$userLabel to ${correctionsFile.absolutePath}")
//        } catch (e: Exception) {
//            Log.e("MainActivity", "Failed to save correction CSV: ${e.message}")
//        }
//         */
//    }

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
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = false
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            // If we had real-time frame processing:
            // if (isTracking) { processFrameWithVideoProcessor() }
        }
    }

    // CameraDevice state callback
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

        // Inflate layout
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Hide advanced processing if not used
        viewBinding.processedFrameView.visibility = View.GONE

        // SharedPreferences
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)

        // Listen for shutter_speed changes
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

        // CameraManager
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

        // The TextureView
        textureView = viewBinding.viewFinder
        switchDisplayMode = findViewById(R.id.switchDisplayMode)

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
                    "Camera & Audio permissions are required.",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }

        // If permissions are already granted
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

        // (Optional) If we wanted to load PyTorch models at startup, from older code:
//        loadAllModelsOnStartup("handwriting_lstm_model.pt", "digits_lstm_model.pt")
    }

    // ------------------------------------------------------------------------------------
    // Start/Stop Tracking => Start/Stop Recording
    // (In older code, we had final predictions, askUserForCorrection, etc.)
    // ------------------------------------------------------------------------------------
    private fun startTracking() {
        isTracking = true
        viewBinding.startTrackingButton.text = "Stop Tracking"
        viewBinding.startTrackingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)

        // If we had a videoProcessor:
        // videoProcessor.clearTrackingData()
        // viewBinding.processedFrameView.visibility = View.VISIBLE

        startRecordingVideo()
        Toast.makeText(this, "Recording started.", Toast.LENGTH_SHORT).show()
    }

    private fun stopTracking() {
        isTracking = false
        viewBinding.startTrackingButton.text = "Start Tracking"
        viewBinding.startTrackingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)

        // If we had a videoProcessor:
        // viewBinding.processedFrameView.visibility = View.GONE
        // viewBinding.processedFrameView.setImageBitmap(null)

        stopRecordingVideo()
        Toast.makeText(this, "Recording stopped.", Toast.LENGTH_SHORT).show()
    }

    // ------------------------------------------------------------------------------------
    // Rolling-shutter recording logic (public Movies folder)
    // ------------------------------------------------------------------------------------
    private fun startRecordingVideo() {
        if (!allPermissionsGranted()) {
            Toast.makeText(this, "Camera/Audio permissions not granted.", Toast.LENGTH_SHORT).show()
            return
        }
        if (cameraDevice == null) {
            Toast.makeText(this, "CameraDevice not ready.", Toast.LENGTH_SHORT).show()
            return
        }

        // Release old recorder if needed
        if (mediaRecorder != null) {
            try { mediaRecorder?.stop() } catch (_: Exception) {}
            mediaRecorder?.reset()
            mediaRecorder?.release()
            mediaRecorder = null
        }

        try {
            mediaRecorder = MediaRecorder().apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setVideoSource(MediaRecorder.VideoSource.SURFACE)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            }

            // Public Movies folder
            @Suppress("DEPRECATION")
            val moviesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES)
            if (moviesDir == null || (!moviesDir.exists() && !moviesDir.mkdirs())) {
                Toast.makeText(this, "Cannot access public Movies folder.", Toast.LENGTH_LONG).show()
                return
            }

            outputFile = File(moviesDir, "Xamera_${System.currentTimeMillis()}.mp4")
            mediaRecorder?.setOutputFile(outputFile!!.absolutePath)

            // For Moto G Pure, 1280x720 is typically safe
            val recordSize = videoSize ?: Size(1280, 720)
            mediaRecorder?.apply {
                setVideoEncoder(MediaRecorder.VideoEncoder.H264)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                setVideoEncodingBitRate(2_000_000)
                setVideoFrameRate(30)
                setVideoSize(recordSize.width, recordSize.height)
                prepare()
            }

            isRecordingVideo = true

            // Build capture session for preview + record
            val texture = textureView.surfaceTexture ?: return
            previewSize?.let { texture.setDefaultBufferSize(it.width, it.height) }
            val previewSurface = Surface(texture)
            val recorderSurface = mediaRecorder!!.surface

            // Use TEMPLATE_RECORD
            captureRequestBuilder = cameraDevice?.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
            captureRequestBuilder?.addTarget(previewSurface)
            captureRequestBuilder?.addTarget(recorderSurface)

            // Apply rolling shutter to the RECORD capture request
            applyRollingShutterForRecording(captureRequestBuilder)

            cameraDevice?.createCaptureSession(
                listOf(previewSurface, recorderSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        cameraCaptureSessions = session
                        try {
                            // Keep manual shutter if user set it
                            captureRequestBuilder?.set(
                                CaptureRequest.CONTROL_MODE,
                                CameraMetadata.CONTROL_MODE_OFF
                            )
                            cameraCaptureSessions?.setRepeatingRequest(
                                captureRequestBuilder!!.build(),
                                null,
                                backgroundHandler
                            )
                            mediaRecorder?.start()
                        } catch (e: CameraAccessException) {
                            Toast.makeText(
                                this@MainActivity,
                                "Failed to start recording: ${e.message}",
                                Toast.LENGTH_LONG
                            ).show()
                            e.printStackTrace()
                        }
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Toast.makeText(
                            this@MainActivity,
                            "Capture session config failed.",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                },
                backgroundHandler
            )

        } catch (e: Exception) {
            Log.e("MainActivity", "MediaRecorder error: ${e.message}", e)
            Toast.makeText(this, "Cannot record: ${e.message}", Toast.LENGTH_LONG).show()
            mediaRecorder?.reset()
            mediaRecorder?.release()
            mediaRecorder = null
            isRecordingVideo = false
        }
    }

    /**
     * Rolling shutter for the RECORD capture request.
     */
    private fun applyRollingShutterForRecording(builder: CaptureRequest.Builder?) {
        if (builder == null) return
        val shutterSetting = sharedPreferences.getString("shutter_speed", "15")?.toInt() ?: 15
        val shutterValue = if (shutterSetting >= 5) 1000000000L / shutterSetting else 0L

        if (shutterValue > 0) {
            builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
            builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterValue)
            Toast.makeText(
                this,
                "Rolling shutter ON for recording at about 1/$shutterSetting s.",
                Toast.LENGTH_LONG
            ).show()
        } else {
            builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            Toast.makeText(
                this,
                "Rolling shutter OFF (auto) for recording.",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    private fun stopRecordingVideo() {
        if (!isRecordingVideo) return
        try {
            mediaRecorder?.stop()
        } catch (e: Exception) {
            Log.e("MainActivity", "Error stopping recording: ${e.message}", e)
        }
        mediaRecorder?.reset()
        mediaRecorder?.release()
        mediaRecorder = null
        isRecordingVideo = false

        // Return to normal preview
        createCameraPreview()

        // Show path if file was created
        outputFile?.let { file ->
            if (file.exists()) {
                Toast.makeText(
                    this,
                    "Video saved:\n${file.absolutePath}",
                    Toast.LENGTH_LONG
                ).show()
            } else {
                Toast.makeText(this, "No output file found.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ------------------------------------------------------------------------------------
    // onActivityResult
    // ------------------------------------------------------------------------------------
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == SETTINGS_REQUEST_CODE && resultCode == RESULT_OK) {
            val shutterSpeedSetting = sharedPreferences.getString("shutter_speed", "60")?.toInt() ?: 60
            shutterSpeed = 1000000000L / shutterSpeedSetting
            Toast.makeText(this, "Settings updated", Toast.LENGTH_SHORT).show()
            updateShutterSpeed()
        }
    }

    // ------------------------------------------------------------------------------------
    // onResume / onPause
    // ------------------------------------------------------------------------------------
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
    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
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

            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?: return

            // For Motorola G Pure, pick 1280x720 or smaller
            previewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture::class.java))
            videoSize = chooseOptimalSize(map.getOutputSizes(MediaRecorder::class.java))

            cameraManager.openCamera(cameraId, stateCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        } catch (e: SecurityException) {
            e.printStackTrace()
            Toast.makeText(this, "Camera permission needed.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun chooseOptimalSize(choices: Array<Size>): Size {
        // For Moto G Pure, 1280×720 is typical. If not found, pick smallest.
        val targetWidth = 1280
        val targetHeight = 720
        val found720p = choices.find { it.width == targetWidth && it.height == targetHeight }
        if (found720p != null) {
            return found720p
        }
        return choices.minByOrNull { it.width * it.height } ?: choices[0]
    }

    private fun getCameraId(): String {
        for (id in cameraManager.cameraIdList) {
            val facing = cameraManager.getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING)
            if (!isFrontCamera && facing == CameraCharacteristics.LENS_FACING_BACK) {
                return id
            } else if (isFrontCamera && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                return id
            }
        }
        return cameraManager.cameraIdList.first()
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

        // If we had a videoProcessor:
        // viewBinding.processedFrameView.visibility = View.GONE
        // viewBinding.processedFrameView.setImageBitmap(null)

        reopenCamera()
    }

    private fun createCameraPreview() {
        try {
            val texture = textureView.surfaceTexture ?: return
            previewSize?.let { texture.setDefaultBufferSize(it.width, it.height) }
            val surface = Surface(texture)

            captureRequestBuilder = cameraDevice?.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            captureRequestBuilder?.addTarget(surface)

            // Rolling shutter for preview
            applyRollingShutter()

            updateShutterSpeed()
            applyFlashIfEnabled()
            applyLightingMode()
            applyZoom()

            cameraDevice?.createCaptureSession(
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
                            "Preview config failed.",
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
            // Keep CONTROL_MODE_OFF if user sets manual shutter
            captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
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
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
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
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterValue)
                Toast.makeText(
                    this@MainActivity,
                    "Rolling shutter ON ~1/$shutterSetting s for preview.",
                    Toast.LENGTH_LONG
                ).show()
            } else {
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                Toast.makeText(
                    this@MainActivity,
                    "Rolling shutter OFF (auto) for preview.",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

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
            try {
                cameraCaptureSessions?.setRepeatingRequest(build(), null, backgroundHandler)
                if (shutterSetting <= 6000) {
                    Toast.makeText(
                        this@MainActivity,
                        "Shutter ~1/$shutterSetting s (preview).",
                        Toast.LENGTH_SHORT
                    ).show()
                } else {
                    Toast.makeText(
                        this@MainActivity,
                        "Shutter speed AUTO (preview).",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            } catch (e: CameraAccessException) {
                e.printStackTrace()
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
}
