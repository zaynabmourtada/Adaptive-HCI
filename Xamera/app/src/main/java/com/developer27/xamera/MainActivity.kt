package com.developer27.xamera

// Standard Android imports.
// Imports from our project.
import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Environment
import android.preference.PreferenceManager
import android.util.Log
import android.util.SparseIntArray
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.developer27.xamera.camera.CameraHelper
import com.developer27.xamera.databinding.ActivityMainBinding
import com.developer27.xamera.openGL2D.OpenGL2DActivity
import com.developer27.xamera.openGL3D.OpenGL3DActivity
import com.developer27.xamera.videoprocessing.ProcessedVideoRecorder
import com.developer27.xamera.videoprocessing.VideoProcessor
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * MainActivity:
 * - Sets up the camera preview and UI controls.
 * - Processes frames via VideoProcessor (which applies OpenCV overlays such as drawn traces).
 * - Records the processed frames to a video file.
 * - When the user stops tracking, the app now prompts the user to enter a name for the tracking data
 *   before saving the drawn line data into a text file.
 */
class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper
    // Global TFLite Interpreter instance.
    private var tfliteInterpreter: Interpreter? = null

    // Custom recorder to save processed video frames.
    private var processedVideoRecorder: ProcessedVideoRecorder? = null

    // VideoProcessor applies detection and drawing overlays.
    private var videoProcessor: VideoProcessor? = null

    // Flags to control the processing/recording states.
    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false

    // Permissions required by the app.
    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    companion object {
        private const val SETTINGS_REQUEST_CODE = 1
        // Mapping for camera rotation.
        private val ORIENTATIONS = SparseIntArray().apply {
            append(Surface.ROTATION_0, 90)
            append(Surface.ROTATION_90, 0)
            append(Surface.ROTATION_180, 270)
            append(Surface.ROTATION_270, 180)
        }
    }

    // TextureView listener for the camera preview.
    private val textureListener = object : TextureView.SurfaceTextureListener {
        @SuppressLint("MissingPermission")
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = false
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            if (isProcessing) {
                processFrameWithVideoProcessor()
            }
        }
    }

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Inflate the layout using view binding.
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager

        // Initialize the helper classes.
        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        videoProcessor = VideoProcessor(this)

        // Hide the processed frame view until processing starts.
        viewBinding.processedFrameView.visibility = View.GONE

        // Register permission launcher for Camera and Audio.
        requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
                val camGranted = permissions[Manifest.permission.CAMERA] ?: false
                val micGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
                if (camGranted && micGranted) {
                    if (viewBinding.viewFinder.isAvailable) {
                        cameraHelper.openCamera()
                    } else {
                        viewBinding.viewFinder.surfaceTextureListener = textureListener
                    }
                } else {
                    Toast.makeText(this, "Camera & Audio permissions are required.", Toast.LENGTH_SHORT).show()
                }
            }

        // Open camera if permissions are granted.
        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) {
                cameraHelper.openCamera()
            } else {
                viewBinding.viewFinder.surfaceTextureListener = textureListener
            }
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        // Set up button click listeners.
        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) {
                stopProcessingAndRecording()
            } else {
                startProcessingAndRecording()
            }
        }
        viewBinding.switchCameraButton.setOnClickListener { switchCamera() }
        viewBinding.twoDOnlyButton.setOnClickListener { launch2DOnlyFeature() }
        viewBinding.threeDOnlyButton.setOnClickListener { launch3DOnlyFeature() }
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        viewBinding.unityButton.setOnClickListener {
            startActivity(Intent(this, com.xamera.ar.core.components.java.sharedcamera.SharedCameraActivity::class.java))
        }

        // Load the TensorFlow Lite model on a separate thread.
        loadBestModelOnStartupThreaded("YOLOv3_float32.tflite")
        cameraHelper.setupZoomControls()
        sharedPreferences.registerOnSharedPreferenceChangeListener { _, key ->
            if (key == "shutter_speed") {
                cameraHelper.updateShutterSpeed()
            }
        }
    }

    // Start the processing and recording session.
    private fun startProcessingAndRecording() {
        isRecording = true
        isProcessing = true
        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE

        // Reset tracking data.
        videoProcessor?.clearTrackingData()

        // Set up the video recorder with a determined output path.
        val outputPath = getProcessedVideoOutputPath()
        processedVideoRecorder = ProcessedVideoRecorder(640, 480, outputPath)
        processedVideoRecorder?.start()

        Toast.makeText(this, "Processing + Recording started.", Toast.LENGTH_SHORT).show()
    }

    // Stop the processing and recording session.
    private fun stopProcessingAndRecording() {
        isRecording = false
        isProcessing = false
        viewBinding.startProcessingButton.text = "Start Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)
        processedVideoRecorder?.stop()
        processedVideoRecorder = null

        // NEW: Prompt the user to enter a name for the tracking data before saving.
        videoProcessor?.promptSaveLineData()

        Toast.makeText(this, "Processing + Recording stopped.", Toast.LENGTH_SHORT).show()
    }

    // Process a frame from the camera preview using VideoProcessor.
    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true

        videoProcessor?.processFrame(bitmap) { processedBitmap ->
            runOnUiThread {
                if (processedBitmap != null && isProcessing) {
                    viewBinding.processedFrameView.setImageBitmap(processedBitmap)
                    processedVideoRecorder?.recordFrame(processedBitmap)
                }
                isProcessingFrame = false
            }
        }
    }

    // Determine an output path for the processed video file.
    private fun getProcessedVideoOutputPath(): String {
        @Suppress("DEPRECATION")
        val moviesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES)
        if (!moviesDir.exists()) {
            moviesDir.mkdirs()
        }
        return File(moviesDir, "Processed_${System.currentTimeMillis()}.mp4").absolutePath
    }

    // Load the best model on a background thread.
    private fun loadBestModelOnStartupThreaded(bestModel: String) {
        Thread {
            val bestLoadedPath = copyAssetModelBlocking(bestModel)
            runOnUiThread {
                if (bestLoadedPath.isNotEmpty()) {
                    try {
                        // Create a GPU delegate and configure the TFLite interpreter.
                        val gpuDelegate = GpuDelegate()
                        val options = Interpreter.Options().apply {
                            addDelegate(gpuDelegate)
                            setNumThreads(Runtime.getRuntime().availableProcessors())
                        }
                        // Load the model as a MappedByteBuffer.
                        tfliteInterpreter = Interpreter(loadMappedFile(bestLoadedPath), options)
                        videoProcessor?.setTFLiteModel(tfliteInterpreter!!)
                        Toast.makeText(this, "TFLite Model loaded: $bestModel", Toast.LENGTH_SHORT).show()
                    } catch (e: Exception) {
                        Toast.makeText(this, "Error loading TFLite model: ${e.message}", Toast.LENGTH_LONG).show()
                        Log.e("MainActivity", "TFLite Interpreter error", e)
                    }
                } else {
                    Toast.makeText(this, "Failed to copy or load $bestModel", Toast.LENGTH_SHORT).show()
                }
            }
        }.start()
    }

    // Load the model file into a MappedByteBuffer.
    private fun loadMappedFile(modelPath: String): MappedByteBuffer {
        val file = File(modelPath)
        val fileInputStream = file.inputStream()
        val fileChannel = fileInputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    // Copy the model from the assets folder to the device's internal storage.
    private fun copyAssetModelBlocking(assetName: String): String {
        return try {
            val outFile = File(filesDir, assetName)
            if (outFile.exists() && outFile.length() > 0) {
                return outFile.absolutePath
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

    // Launch the 2D-only feature activity.
    private fun launch2DOnlyFeature() {
        try {
            startActivity(Intent(this, OpenGL2DActivity::class.java))
        } catch (e: Exception) {
            Toast.makeText(this, "Error launching 2D feature: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    // Launch the 3D-only feature activity.
    private fun launch3DOnlyFeature() {
        try {
            val intent = Intent(this, OpenGL3DActivity::class.java)
            startActivity(intent)
        } catch (e: Exception) {
            Toast.makeText(this, "Error launching 3D feature: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onResume() {
        super.onResume()
        cameraHelper.startBackgroundThread()
        if (viewBinding.viewFinder.isAvailable) {
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
            } else {
                requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
            }
        } else {
            viewBinding.viewFinder.surfaceTextureListener = textureListener
        }
    }

    override fun onPause() {
        if (isRecording) {
            stopProcessingAndRecording()
        }
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
        super.onPause()
    }

    // Check that all required permissions are granted.
    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    // Switch between front and back cameras.
    private var isFrontCamera = false
    private fun switchCamera() {
        if (isRecording) {
            stopProcessingAndRecording()
        }
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        cameraHelper.openCamera()
    }
}
