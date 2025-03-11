package com.developer27.xamera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.preference.PreferenceManager
import android.util.Log
import android.util.SparseIntArray
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import com.developer27.xamera.camera.CameraHelper
import com.developer27.xamera.databinding.ActivityMainBinding
import com.developer27.xamera.videoprocessing.ProcessedFrameRecorder
import com.developer27.xamera.videoprocessing.ProcessedVideoRecorder
import com.developer27.xamera.videoprocessing.Settings
import com.developer27.xamera.videoprocessing.VideoProcessor
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper
    private var tfliteInterpreter: Interpreter? = null
    private var processedVideoRecorder: ProcessedVideoRecorder? = null
    private var processedFrameRecorder: ProcessedFrameRecorder? = null
    private var videoProcessor: VideoProcessor? = null

    // Flag for tracking (start/stop tracking mode)
    private var isRecording = false
    // Flag for frame processing
    private var isProcessing = false
    private var isProcessingFrame = false

    // Variable for inference result.
    private var inferenceResult = ""

    // Stores the tracking coordinates.
    private var trackingCoordinates: String = ""

    // For toggling digit/letter recognition.
    var isLetterSelected = true
    var isDigitSelected = !isLetterSelected

    // New flag for writing mode.
    private var isWriting = false

    // New flag to clear prediction when returning from an external intent.
    private var shouldClearPrediction = false

    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    companion object {
        private const val SETTINGS_REQUEST_CODE = 1
        private val ORIENTATIONS = SparseIntArray().apply {
            append(Surface.ROTATION_0, 90)
            append(Surface.ROTATION_90, 0)
            append(Surface.ROTATION_180, 270)
            append(Surface.ROTATION_270, 180)
        }
    }

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
        // Prevent screen from turning off
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Lock screen orientation to portrait (choose as needed)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT

        // Install the splash screen (Android 12+)
        installSplashScreen()
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager

        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        videoProcessor = VideoProcessor(this)

        // Hide the processed frame view initially.
        viewBinding.processedFrameView.visibility = View.GONE

        // Set default text for the predicted letter TextView.
        viewBinding.predictedLetterTextView.text = "No Prediction Yet"

        // When the title container is clicked, open the URL in a browser.
        viewBinding.titleContainer.setOnClickListener {
            val url = "https://www.zhangxiao.me/"
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        }

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

        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) {
                cameraHelper.openCamera()
            } else {
                viewBinding.viewFinder.surfaceTextureListener = textureListener
            }
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        // Set up the Start Tracking button.
        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) {
                // Stop tracking and update prediction.
                stopProcessingAndRecording()
            } else {
                startProcessingAndRecording()
            }
        }

        // Set up the Switch Camera, About, and Settings buttons.
        viewBinding.switchCameraButton.setOnClickListener { switchCamera() }
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        viewBinding.radioToggle.setTextColor(android.graphics.Color.parseColor("#FFCB05"))
        viewBinding.radioToggle.buttonTintList = android.content.res.ColorStateList.valueOf(android.graphics.Color.parseColor("#FFCB05"))
        viewBinding.radioToggle.text = if (isLetterSelected) "Letter" else "Digit"
        viewBinding.radioToggle.isChecked = isLetterSelected
        viewBinding.radioToggle.setOnClickListener {
            isLetterSelected = !isLetterSelected
            isDigitSelected = !isLetterSelected
            viewBinding.radioToggle.text = if (isLetterSelected) "Letter" else "Digit"
            viewBinding.radioToggle.isChecked = isLetterSelected
        }

        // Set up the Start Writing button.
        viewBinding.startWritingButton.setOnClickListener {
            toggleWritingMode()
        }

        // Set up the Clear Prediction button ("C") under the zoom buttons.
        viewBinding.clearPredictionButton.setOnClickListener {
            // Stop writing if active.
            if (isWriting) {
                isWriting = false
                viewBinding.startWritingButton.text = "Start Writing"
                viewBinding.startWritingButton.backgroundTintList =
                    ContextCompat.getColorStateList(this, R.color.green)
            }
            // Stop tracking if active.
            if (isRecording) {
                stopProcessingAndRecording()
            }
            // Reset the prediction panel.
            viewBinding.predictedLetterTextView.text = "No Prediction Yet"
        }

        // Load ML models.
        loadTFLiteModelOnStartupThreaded("YOLOv3_float32.tflite")
        loadTFLiteModelOnStartupThreaded("DigitRecog_float32.tflite")

        cameraHelper.setupZoomControls()
        sharedPreferences.registerOnSharedPreferenceChangeListener { _, key ->
            if (key == "shutter_speed") {
                cameraHelper.updateShutterSpeed()
            }
        }
    }

    // Function for toggling writing mode.
    private fun toggleWritingMode() {
        if (!isWriting) {
            // Start writing mode.
            isWriting = true
            viewBinding.startWritingButton.text = "Stop Writing"
            viewBinding.startWritingButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.red)
            // Clear the prediction panel.
            viewBinding.predictedLetterTextView.text = ""
        } else {
            // Stop writing mode.
            isWriting = false
            viewBinding.startWritingButton.text = "Start Writing"
            viewBinding.startWritingButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.green)
            // When writing stops, check the accumulated prediction.
            val prediction = viewBinding.predictedLetterTextView.text.toString()
            if (isLetterSelected) {
                // Letter mode: ask if user wants to send an email.
                AlertDialog.Builder(this)
                    .setTitle("Send Email")
                    .setMessage("Do you wish to send an email with the text: $prediction?")
                    .setPositiveButton("Yes") { _, _ ->
                        sendEmail(prediction)
                    }
                    .setNegativeButton("No") { _, _ ->
                        launch3DActivity()
                    }
                    .show()
            } else {
                // Digit mode: if the prediction is composed solely of digits, ask if user wants to call.
                if (prediction.matches(Regex("\\d+"))) {
                    AlertDialog.Builder(this)
                        .setTitle("Call Number")
                        .setMessage("Do you wish to call the number $prediction?")
                        .setPositiveButton("Yes") { _, _ ->
                            makePhoneCall(prediction)
                        }
                        .setNegativeButton("No") { _, _ ->
                            launch3DActivity()
                        }
                        .show()
                } else {
                    launch3DActivity()
                }
            }
        }
    }

    // New function to send an email with the prediction.
    private fun sendEmail(text: String) {
        val emailIntent = Intent(Intent.ACTION_SENDTO).apply {
            data = Uri.parse("mailto:") // Only email apps should handle this.
            putExtra(Intent.EXTRA_SUBJECT, "Air-Written Email by Xamera")
            putExtra(Intent.EXTRA_TEXT, text)
        }
        shouldClearPrediction = true
        startActivity(emailIntent)
    }

    private fun startProcessingAndRecording() {
        isRecording = true
        isProcessing = true
        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE

        videoProcessor?.reset()

        if (Settings.ExportData.videoDATA) {
            val dims = videoProcessor?.getModelDimensions()
            val width = dims?.first ?: 416
            val height = dims?.second ?: 416
            val outputPath = getProcessedVideoOutputPath()
            processedVideoRecorder = ProcessedVideoRecorder(width, height, outputPath)
            processedVideoRecorder?.start()
        }
    }

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

        val outputPath = getProcessedImageOutputPath()
        processedFrameRecorder = ProcessedFrameRecorder(outputPath)
        with(Settings.ExportData) {
            if (frameIMG) {
                val bitmap = videoProcessor?.exportTraceForInference()
                if (bitmap != null) {
                    processedFrameRecorder?.save(bitmap)
                }
            }
        }

        // Compute the inference result.
        initializeInferenceResult()
        // If in writing mode, append the prediction to the TextView.
        if (isWriting) {
            val currentText = viewBinding.predictedLetterTextView.text.toString()
            val newText = if (currentText == "No Prediction Available Yet") {
                inferenceResult
            } else {
                currentText + inferenceResult
            }
            viewBinding.predictedLetterTextView.text = newText
        } else {
            // Otherwise, just update the panel.
            viewBinding.predictedLetterTextView.text = inferenceResult
        }

        // Retrieve tracking coordinates.
        trackingCoordinates = videoProcessor?.getTrackingCoordinatesString() ?: ""
    }

    private fun initializeInferenceResult() {
        if (isLetterSelected) {
            inferenceResult = "ML - Inference: Letters"
        } else if (isDigitSelected) {
            inferenceResult = runDigitRecognitionInference()
        }
    }

    private fun runDigitRecognitionInference(): String {
        val digitBitmap = videoProcessor?.exportTraceForInference()
        if (digitBitmap == null) {
            Log.e("MainActivity", "No digit image available for inference")
            return "Error"
        }
        val grayBitmap = convertToGrayscale(digitBitmap)
        val inputBuffer = convertBitmapToGrayscaleByteBuffer(grayBitmap)
        val outputArray = Array(1) { FloatArray(10) }
        if (tfliteInterpreter == null) {
            Log.e("MainActivity", "Digit model interpreter not set")
            return "Error"
        }
        tfliteInterpreter?.run(inputBuffer, outputArray)
        val predictedDigit = outputArray[0].indices.maxByOrNull { outputArray[0][it] } ?: -1
        Log.d("MainActivity", "Digit model predicted: $predictedDigit")
        return predictedDigit.toString()
    }

    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val grayscaleBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscaleBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix().apply { setSaturation(0f) }
        val filter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = filter
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
        return grayscaleBitmap
    }

    private fun convertBitmapToGrayscaleByteBuffer(bitmap: Bitmap): ByteBuffer {
        val inputSize = bitmap.width * bitmap.height
        val byteBuffer = ByteBuffer.allocateDirect(inputSize * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (pixel in intValues) {
            val r = (pixel shr 16 and 0xFF).toFloat()
            val normalized = r / 255.0f
            byteBuffer.putFloat(normalized)
        }
        return byteBuffer
    }

    private fun launch3DActivity() {
        // Send the accumulated prediction information to the 3D launcher.
        val intent = Intent(this, com.xamera.ar.core.components.java.sharedcamera.SharedCameraActivity::class.java)
        intent.putExtra("LETTER_KEY", viewBinding.predictedLetterTextView.text.toString())
        val pathCoordinates = if (trackingCoordinates.isNotEmpty()) {
            trackingCoordinates
        } else {
            "0.0,0.0,0.0;5.0,10.0,-5.0;-5.0,15.0,10.0;20.0,-5.0,5.0;-10.0,0.0,-10.0;10.0,-15.0,15.0;0.0,20.0,-5.0"
        }
        intent.putExtra("PATH_COORDINATES", pathCoordinates)
        shouldClearPrediction = true
        startActivity(intent)
    }

    // Simulate making a phone call using ACTION_DIAL.
    private fun makePhoneCall(digits: String) {
        val callIntent = Intent(Intent.ACTION_DIAL)
        callIntent.data = Uri.parse("tel:$digits")
        shouldClearPrediction = true
        startActivity(callIntent)
    }

    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true
        videoProcessor?.processFrame(bitmap) { processedFrames ->
            runOnUiThread {
                processedFrames?.let { (outputBitmap, preprocessedBitmap) ->
                    if (isProcessing) {
                        viewBinding.processedFrameView.setImageBitmap(outputBitmap)
                        with(Settings.ExportData) {
                            if (videoDATA) {
                                processedVideoRecorder?.recordFrame(preprocessedBitmap)
                            }
                        }
                    }
                }
                isProcessingFrame = false
            }
        }
    }

    private fun getProcessedVideoOutputPath(): String {
        @Suppress("DEPRECATION")
        val moviesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES)
        if (!moviesDir.exists()) {
            moviesDir.mkdirs()
        }
        return File(moviesDir, "Processed_${System.currentTimeMillis()}.mp4").absolutePath
    }

    private fun getProcessedImageOutputPath(): String {
        @Suppress("DEPRECATION")
        val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        if (!picturesDir.exists()) {
            picturesDir.mkdirs()
        }
        return File(picturesDir, "Processed_${System.currentTimeMillis()}.jpg").absolutePath
    }

    private fun loadTFLiteModelOnStartupThreaded(modelName: String) {
        Thread {
            val bestLoadedPath = copyAssetModelBlocking(modelName)
            runOnUiThread {
                if (bestLoadedPath.isNotEmpty()) {
                    try {
                        val options = Interpreter.Options().apply {
                            setNumThreads(Runtime.getRuntime().availableProcessors())
                        }
                        var delegateAdded = false
                        try {
                            val nnApiDelegate = NnApiDelegate()
                            options.addDelegate(nnApiDelegate)
                            delegateAdded = true
                            Log.d("MainActivity", "NNAPI delegate added successfully.")
                        } catch (e: Exception) {
                            Log.d("MainActivity", "NNAPI delegate unavailable, falling back to GPU delegate.", e)
                        }
                        if (!delegateAdded) {
                            try {
                                val gpuDelegate = GpuDelegate()
                                options.addDelegate(gpuDelegate)
                                Log.d("MainActivity", "GPU delegate added successfully.")
                            } catch (e: Exception) {
                                Log.d("MainActivity", "GPU delegate unavailable, will use CPU only.", e)
                            }
                        }
                        when (modelName) {
                            "YOLOv3_float32.tflite" -> {
                                videoProcessor?.setInterpreter(Interpreter(loadMappedFile(bestLoadedPath), options))
                            }
                            "DigitRecog_float32.tflite" -> {
                                tfliteInterpreter = Interpreter(loadMappedFile(bestLoadedPath), options)
                            }
                            else -> Log.d("MainActivity", "No model processing method defined for $modelName")
                        }
                    } catch (e: Exception) {
                        Toast.makeText(this, "Error loading TFLite model: ${e.message}", Toast.LENGTH_LONG).show()
                        Log.d("MainActivity", "TFLite Interpreter error", e)
                    }
                } else {
                    Toast.makeText(this, "Failed to copy or load $modelName", Toast.LENGTH_SHORT).show()
                }
            }
        }.start()
    }

    private fun loadMappedFile(modelPath: String): MappedByteBuffer {
        val file = File(modelPath)
        val fileInputStream = file.inputStream()
        val fileChannel = fileInputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

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
        // When returning from an external intent, clear the prediction.
        if (shouldClearPrediction) {
            viewBinding.predictedLetterTextView.text = "No Prediction Yet"
            shouldClearPrediction = false
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

    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }
}