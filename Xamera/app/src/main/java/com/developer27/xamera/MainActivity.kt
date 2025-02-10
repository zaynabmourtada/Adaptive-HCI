package com.developer27.xamera

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
import android.widget.EditText
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.developer27.xamera.camera.CameraHelper
import com.developer27.xamera.databinding.ActivityMainBinding
import com.developer27.xamera.openGL2D.OpenGL2DActivity
import com.developer27.xamera.openGL3D.OpenGL3DActivity
import com.developer27.xamera.videoprocessing.ProcessedFrameRecorder
import com.developer27.xamera.videoprocessing.ProcessedVideoRecorder
import com.developer27.xamera.videoprocessing.Settings
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
 * - When the user stops tracking, the app first prompts the user to name the tracking (line) data file
 *   (which is saved in Documents/tracking) and then prompts the user to name a second file (which is saved
 *   in Documents/2d_letter with the content of the hardcoded alphabet).
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
    private var processedFrameRecorder: ProcessedFrameRecorder? = null

    // VideoProcessor applies detection and drawing overlays.
    private var videoProcessor: VideoProcessor? = null

    // Flags to control the processing/recording states.
    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false

    //This variable is for inference result
    private var inferenceResult = ""

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

        // Initialize helper classes.
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

        // Retrieve input tensor shape from the TFLite interpreter.
        val inputTensor = tfliteInterpreter?.getInputTensor(0)
        val inputShape = inputTensor?.shape()

        // Determine width and height based on the export mode.
        val (width, height) = when (Settings.ExportData.current) {
            Settings.ExportData.Mode.SCREEN -> {
                // Use cameraHelper.previewSize if available; otherwise fallback to 416x416.
                cameraHelper.videoSize?.let { it.height to it.width } ?: (1920 to 1080)
            }
            Settings.ExportData.Mode.MODEL -> {
                // Use the model's input tensor shape: [1, height, width, channels] is assumed.
                (inputShape?.getOrNull(2) ?: 416) to (inputShape?.getOrNull(1) ?: 416)
            }
        }

        // Save video
        with(Settings.ExportData) {
            if (videoDATA) {
                val outputPath = getProcessedVideoOutputPath()
                processedVideoRecorder = ProcessedVideoRecorder(width, height, outputPath)
                processedVideoRecorder?.start()
            }
        }
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

        // Save a processed frame as a jpg for testing.
        val outputPath = getProcessedImageOutputPath()
        processedFrameRecorder = ProcessedFrameRecorder(outputPath)
        with(Settings.ExportData) {
            if (frameIMG) {
                val bitmap = videoProcessor?.exportTraceForInference()
                if (bitmap != null) { processedFrameRecorder?.save(bitmap) }
            }
        }

        // TODO - Zaynab: Call inference result initialization function
        intializeInferenceResult()

        // First: Prompt to save the tracking (line) data in Documents/tracking.
        videoProcessor?.promptSaveLineData()

        // Then, after a delay, prompt to save the Letter Inference Data.
        promptSaveLetterInferenceData()
    }

    // TODO- Zaynab: Intiliaze a logic to initialize the inference from Machine Learning Model
    private fun intializeInferenceResult(){
        inferenceResult = "ML - Inference"; // "ML - Inference" is a place holder function
    }

    /**
     * Prompts the user to enter a file name for saving the Letter Inference Data.
     * This displays an AlertDialog where the user can input the desired file name.
     *
     * When the user confirms, it calls [saveLetterInferenceData] to write the file.
     */
    fun promptSaveLetterInferenceData() {
        // Create an EditText for the file name input.
        val editText = EditText(this).apply {
            hint = "Enter file name"
        }

        // Build and display an AlertDialog to prompt for the file name.
        AlertDialog.Builder(this)
            .setTitle("Save Letter Inference Data")
            .setMessage("Enter a file name for the Letter Inference Data:")
            .setView(editText)
            .setPositiveButton("Save") { _, _ ->
                val fileName = editText.text.toString().trim()
                if (fileName.isNotEmpty()) {
                    // If the file name is provided, save the Letter Inference Data file.
                    saveLetterInferenceData(fileName)
                } else {
                    Toast.makeText(this, "File name cannot be empty.", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel") { dialog, _ ->
                dialog.dismiss()
            }
            .setCancelable(false)
            .show()
    }

    /**
     * Saves a text file containing a hardcoded alphabet into the Documents/2d_letter folder.
     *
     * This function uses the public Documents directory (similar to saveLineDataToFile())
     * but creates/uses a subdirectory named "2d_letter". The file name is built using the
     * user-provided base name and the current timestamp.
     *
     * @param userDataName The base name provided by the user.
     */
    private fun saveLetterInferenceData(userDataName: String) {
        try {
            // Get the public Documents directory.
            val documentsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            // Create a subfolder named "2d_letter" within Documents.
            val letterDir = File(documentsDir, "2d_letter")
            if (!letterDir.exists()) {
                letterDir.mkdirs()
            }
            // Create a unique file name using the user-provided name and the current timestamp.
            val fileName = "${userDataName}_letter_${System.currentTimeMillis()}.txt"
            val file = File(letterDir, fileName)

            // Define the hardcoded alphabet string.
            val alphabetData = inferenceResult
            // Write the alphabet string into the file.
            file.writeText(alphabetData)

            // Notify the user that the file was saved successfully.
            Toast.makeText(this, "Letter inference data saved. Check Documents/2d_letter.", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Error saving Letter inference data", Toast.LENGTH_SHORT).show()
        }
    }

    // Process a frame from the camera preview using VideoProcessor.
    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true
        videoProcessor?.processFrame(bitmap) { processedFrames ->
            runOnUiThread {
                processedFrames?.let { (outputBitmap, videoBitmap) ->
                    if (isProcessing) {
                        viewBinding.processedFrameView.setImageBitmap(outputBitmap)
                        with(Settings.ExportData) {
                            if (videoDATA) {
                                processedVideoRecorder?.recordFrame(outputBitmap)
                            }
                        }
                    }
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

    // Determine an output path for the processed image file.
    private fun getProcessedImageOutputPath(): String {
        @Suppress("DEPRECATION")
        val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        if (!picturesDir.exists()) {
            picturesDir.mkdirs()
        }
        return File(picturesDir, "Processed_${System.currentTimeMillis()}.jpg").absolutePath
    }

    // Load the best model on a background thread.
    private fun loadBestModelOnStartupThreaded(bestModel: String) {
        Thread {
            val bestLoadedPath = copyAssetModelBlocking(bestModel)
            runOnUiThread {
                if (bestLoadedPath.isNotEmpty()) {
                    try {
                        val gpuDelegate = GpuDelegate()
                        val options = Interpreter.Options().apply {
                            addDelegate(gpuDelegate)
                            setNumThreads(Runtime.getRuntime().availableProcessors())
                        }
                        tfliteInterpreter = Interpreter(loadMappedFile(bestLoadedPath), options)
                        videoProcessor?.setTFLiteModel(tfliteInterpreter!!)
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
            startActivity(Intent(this, OpenGL3DActivity::class.java))
        } catch (e: Exception) {
            Toast.makeText(this, "Error launching 3D feature: ${e.message}", Toast.LENGTH_SHORT).show()
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
}
