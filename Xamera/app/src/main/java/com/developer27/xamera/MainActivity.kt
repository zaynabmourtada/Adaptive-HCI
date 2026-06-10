package com.developer27.xamera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import androidx.preference.PreferenceManager
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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper
    private var tfliteInterpreter: Interpreter? = null
    private var letterInterpreter: Interpreter? = null

    private var processedVideoRecorder: ProcessedVideoRecorder? = null
    private var processedFrameRecorder: ProcessedFrameRecorder? = null
    private var videoProcessor: VideoProcessor? = null

    private val mainScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    private var isRecording = false
    private var isProcessing = false
    private var isProcessingFrame = false
    private var trackingCoordinates: String = ""
    var isLetterSelected = true
    private var isWriting = false
    private var shouldClearPrediction = false
    private val accumulatedCoordinates = mutableListOf<String>()
    private var isResetting = false

    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

    companion object {
        private const val SETTINGS_REQUEST_CODE = 1
        private val MIXED_ALPHA_DIGIT_REGEX = Regex("^(?=.*[A-Za-z])(?=.*\\d).+$")
        private val DIGITS_ONLY_REGEX = Regex("\\d+")
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
            if (allPermissionsGranted()) cameraHelper.openCamera()
            else requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = false
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            if (isProcessing) processFrameWithVideoProcessor()
        }
    }

    @SuppressLint("MissingPermission")
    override fun onCreate(savedInstanceState: Bundle?) {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        installSplashScreen()
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        videoProcessor = VideoProcessor()

        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.predictedLetterTextView.text = "No Prediction Yet"

        viewBinding.titleContainer.setOnClickListener {
            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://www.zhangxiao.me/")))
        }

        requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            val camGranted = permissions[Manifest.permission.CAMERA] ?: false
            val micGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
            if (camGranted && micGranted) {
                if (viewBinding.viewFinder.isAvailable) cameraHelper.openCamera()
                else viewBinding.viewFinder.surfaceTextureListener = textureListener
            } else {
                Toast.makeText(this, "Camera & Audio permissions are required.", Toast.LENGTH_SHORT).show()
            }
        }

        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) cameraHelper.openCamera()
            else viewBinding.viewFinder.surfaceTextureListener = textureListener
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) stopProcessingAndRecording() else startProcessingAndRecording()
        }
        viewBinding.switchCameraButton.setOnClickListener { switchCamera() }
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        val letterDigitSwitch = viewBinding.letterDigitSwitch
        updateSwitchStyle(isLetterSelected)
        letterDigitSwitch.isChecked = isLetterSelected
        letterDigitSwitch.setOnCheckedChangeListener { _, isChecked ->
            isLetterSelected = isChecked
            updateSwitchStyle(isLetterSelected)
        }

        viewBinding.startWritingButton.setOnClickListener { toggleWritingMode() }

        viewBinding.clearPredictionButton.setOnClickListener {
            if (isWriting) {
                isWriting = false
                viewBinding.startWritingButton.text = "Start Writing"
                viewBinding.startWritingButton.backgroundTintList =
                    ContextCompat.getColorStateList(this, R.color.green)
            }
            if (isRecording) stopProcessingAndRecording()
            isResetting = true
            viewBinding.predictedLetterTextView.text = "No Prediction Yet"
            accumulatedCoordinates.clear()
            trackingCoordinates = ""
            isResetting = false
        }

        loadTFLiteModelOnStartupThreaded("YOLOv3_float32.tflite")
        loadTFLiteModelOnStartupThreaded("DigitRecog_float32.tflite")
        loadTFLiteModelOnStartupThreaded("LetterRecog_float32.tflite")

        cameraHelper.setupZoomControls()
        sharedPreferences.registerOnSharedPreferenceChangeListener { _, key ->
            if (key == "shutter_speed") cameraHelper.updateShutterSpeed()
        }
    }

    private fun updateSwitchStyle(isLetter: Boolean) {
        val color = if (isLetter) Color.parseColor("#FFCB05") else Color.parseColor("#FFFFFF")
        val colorList = android.content.res.ColorStateList.valueOf(color)
        viewBinding.letterDigitSwitch.apply {
            setTextColor(color)
            thumbTintList = colorList
            trackTintList = colorList
            text = if (isLetter) "Letter" else "Digit"
        }
    }

    private fun toggleWritingMode() {
        if (!isWriting) {
            isResetting = true
            viewBinding.predictedLetterTextView.text = ""
            accumulatedCoordinates.clear()
            trackingCoordinates = ""
            isResetting = false

            isWriting = true
            viewBinding.startWritingButton.text = "Stop Writing"
            viewBinding.startWritingButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.red)
        } else {
            isWriting = false
            viewBinding.startWritingButton.text = "Start Writing"
            viewBinding.startWritingButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.green)
            val prediction = viewBinding.predictedLetterTextView.text.toString()
            if (prediction.matches(MIXED_ALPHA_DIGIT_REGEX) || isLetterSelected) {
                AlertDialog.Builder(this)
                    .setTitle("Send Email")
                    .setMessage("Do you wish to send an email with the text: $prediction?")
                    .setPositiveButton("Yes") { _, _ -> sendEmail(prediction) }
                    .setNegativeButton("No") { _, _ -> launch3DActivity() }
                    .show()
            } else {
                if (prediction.matches(DIGITS_ONLY_REGEX)) {
                    AlertDialog.Builder(this)
                        .setTitle("Call Number")
                        .setMessage("Do you wish to call the number $prediction?")
                        .setPositiveButton("Yes") { _, _ -> makePhoneCall(prediction) }
                        .setNegativeButton("No") { _, _ -> launch3DActivity() }
                        .show()
                } else {
                    launch3DActivity()
                }
            }
        }
    }

    private fun sendEmail(text: String) {
        val intent = Intent(Intent.ACTION_SENDTO).apply {
            data = Uri.parse("mailto:")
            putExtra(Intent.EXTRA_SUBJECT, "Air-Written Email by Xamera")
            putExtra(Intent.EXTRA_TEXT, text)
        }
        shouldClearPrediction = true
        startActivity(intent)
    }

    private fun startProcessingAndRecording() {
        isRecording = true
        isProcessing = true
        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)
        viewBinding.processedFrameView.visibility = View.VISIBLE

        videoProcessor?.reset()
        Toast.makeText(this, "Tracking started", Toast.LENGTH_SHORT).show()

        if (Settings.ExportData.videoDATA) {
            val dims = videoProcessor?.getModelDimensions()
            val outputPath = ProcessedVideoRecorder.getExportedVideoOutputPath()
            processedVideoRecorder = ProcessedVideoRecorder(dims?.inputWidth ?: 416, dims?.inputHeight ?: 416, outputPath)
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

        val processor = videoProcessor ?: return
        val isLetter = isLetterSelected
        val interpreter = if (isLetter) letterInterpreter else tfliteInterpreter
        val writing = isWriting

        mainScope.launch {
            // Compute the trace bitmap once — used for both export and inference
            val traceBitmap = withContext(Dispatchers.Default) { processor.exportTraceForInference() }

            if (Settings.ExportData.frameIMG) {
                withContext(Dispatchers.IO) {
                    ProcessedFrameRecorder(get28x28OutputPath()).save(traceBitmap)
                }
            }

            val result: String? = if (interpreter != null) {
                withContext(Dispatchers.Default) {
                    val outputSize = if (isLetter) 26 else 10
                    val label: (Int) -> String = if (isLetter) { i -> ('A' + i).toString() } else Int::toString
                    runInference(interpreter, traceBitmap, outputSize, label)
                }
            } else null

            val displayResult = result ?: "?"
            val coords = processor.getTrackingCoordinatesString()
            trackingCoordinates = coords

            if (writing) {
                if (coords.isNotEmpty()) accumulateCoordinates(coords)
                val currentText = viewBinding.predictedLetterTextView.text.toString()
                viewBinding.predictedLetterTextView.text =
                    if (currentText == "No Prediction Available Yet") displayResult
                    else currentText + displayResult
            } else {
                viewBinding.predictedLetterTextView.text = displayResult
            }
        }
    }

    private fun accumulateCoordinates(newCoords: String) {
        if (newCoords.isEmpty()) return
        if (accumulatedCoordinates.isEmpty()) {
            accumulatedCoordinates.add(newCoords)
            return
        }
        var offsetX = 0.0
        for (coordStr in accumulatedCoordinates) {
            val maxX = coordStr.split(";")
                .mapNotNull { it.split(",").getOrNull(0)?.toDoubleOrNull() }
                .maxOrNull() ?: 0.0
            offsetX = max(offsetX, maxX)
        }
        offsetX += 10.0
        val adjusted = newCoords.split(";").mapNotNull { pointStr ->
            val parts = pointStr.split(",")
            if (parts.size >= 2) {
                val x = parts[0].toDoubleOrNull() ?: 0.0
                val y = parts[1]
                val z = parts.getOrElse(2) { "0.0" }
                "${x + offsetX},$y,$z"
            } else null
        }
        accumulatedCoordinates.add(adjusted.joinToString(";"))
    }

    private fun get28x28OutputPath(): String {
        @Suppress("DEPRECATION")
        val picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        val dir = File(picturesDir, "Exported Lines from Xamera").also { it.mkdirs() }
        return File(dir, "DrawnLine_28x28_${System.currentTimeMillis()}.png").absolutePath
    }

    private fun runInference(interpreter: Interpreter, bitmap: Bitmap, outputSize: Int, indexToLabel: (Int) -> String): String? {
        return try {
            val buffer = convertBitmapToGrayscaleByteBuffer(convertToGrayscale(bitmap))
            val output = Array(1) { FloatArray(outputSize) }
            interpreter.run(buffer, output)
            val idx = output[0].indices.maxByOrNull { output[0][it] } ?: return null
            indexToLabel(idx)
        } catch (e: Exception) {
            Log.e("MainActivity", "Inference failed: ${e.message}")
            null
        }
    }

    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val out = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.ARGB_8888)
        val paint = Paint().apply {
            colorFilter = ColorMatrixColorFilter(ColorMatrix().apply { setSaturation(0f) })
        }
        Canvas(out).drawBitmap(bitmap, 0f, 0f, paint)
        return out
    }

    private fun convertBitmapToGrayscaleByteBuffer(bitmap: Bitmap): ByteBuffer {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        return ByteBuffer.allocateDirect(pixels.size * 4).apply {
            order(ByteOrder.nativeOrder())
            for (pixel in pixels) {
                putFloat(((pixel shr 16) and 0xFF) / 255.0f)
            }
        }
    }

    private fun launch3DActivity() {
        val coords = when {
            accumulatedCoordinates.isNotEmpty() -> accumulatedCoordinates.joinToString("|")
            trackingCoordinates.isNotEmpty() -> trackingCoordinates
            else -> "0.0,0.0,0.0;5.0,10.0,-5.0;-5.0,15.0,10.0;20.0,-5.0,5.0;-10.0,0.0,-10.0;10.0,-15.0,15.0;0.0,20.0,-5.0"
        }
        val intent = Intent(this, com.xamera.ar.core.components.java.sharedcamera.SharedCameraActivity::class.java).apply {
            putExtra("LETTER_KEY", viewBinding.predictedLetterTextView.text.toString())
            putExtra("PATH_COORDINATES", coords)
        }
        shouldClearPrediction = true
        startActivity(intent)
    }

    private fun makePhoneCall(digits: String) {
        shouldClearPrediction = true
        startActivity(Intent(Intent.ACTION_DIAL, Uri.parse("tel:$digits")))
    }

    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true
        videoProcessor?.processFrame(bitmap) { processedFrames ->
            runOnUiThread {
                if (isResetting) { isProcessingFrame = false; return@runOnUiThread }
                processedFrames?.let { (outputBitmap, preprocessedBitmap) ->
                    if (isProcessing) {
                        viewBinding.processedFrameView.setImageBitmap(outputBitmap)
                        if (Settings.ExportData.videoDATA) {
                            processedVideoRecorder?.recordFrame(preprocessedBitmap)
                        }
                    }
                }
                if (videoProcessor?.getTrackingCoordinatesString().isNullOrEmpty()) resetScreen()
                isProcessingFrame = false
            }
        }
    }

    private fun resetScreen() {
        isResetting = true
        viewBinding.processedFrameView.setImageBitmap(null)
        viewBinding.predictedLetterTextView.text = "No Prediction Yet"
        trackingCoordinates = ""
        isResetting = false
    }

    private fun loadTFLiteModelOnStartupThreaded(modelName: String) {
        Thread {
            val modelPath = copyAssetModelBlocking(modelName)
            runOnUiThread {
                if (modelPath.isEmpty()) {
                    Toast.makeText(this, "Failed to copy or load $modelName", Toast.LENGTH_SHORT).show()
                    return@runOnUiThread
                }
                try {
                    val options = Interpreter.Options().apply {
                        setNumThreads(Runtime.getRuntime().availableProcessors())
                    }
                    var delegateAdded = false
                    try {
                        options.addDelegate(NnApiDelegate())
                        delegateAdded = true
                        Log.d("MainActivity", "NNAPI delegate added.")
                    } catch (e: Exception) {
                        Log.d("MainActivity", "NNAPI unavailable, trying GPU.", e)
                    }
                    if (!delegateAdded) {
                        try {
                            options.addDelegate(GpuDelegate())
                            Log.d("MainActivity", "GPU delegate added.")
                        } catch (e: Exception) {
                            Log.d("MainActivity", "GPU unavailable, using CPU.", e)
                        }
                    }
                    val interpreter = Interpreter(loadMappedFile(modelPath), options)
                    when (modelName) {
                        "YOLOv3_float32.tflite" -> videoProcessor?.setInterpreter(interpreter)
                        "DigitRecog_float32.tflite" -> tfliteInterpreter = interpreter
                        "LetterRecog_float32.tflite" -> letterInterpreter = interpreter
                        else -> Log.d("MainActivity", "No handler for $modelName")
                    }
                } catch (e: Exception) {
                    Toast.makeText(this, "Error loading model: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    private fun loadMappedFile(modelPath: String): MappedByteBuffer {
        val file = File(modelPath)
        return file.inputStream().channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    private fun copyAssetModelBlocking(assetName: String): String {
        return try {
            val outFile = File(filesDir, assetName)
            if (outFile.exists() && outFile.length() > 0) return outFile.absolutePath
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
        if (isRecording) stopProcessingAndRecording()
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        cameraHelper.openCamera()
    }

    override fun onResume() {
        super.onResume()
        accumulatedCoordinates.clear()
        cameraHelper.startBackgroundThread()
        if (viewBinding.viewFinder.isAvailable) {
            if (allPermissionsGranted()) cameraHelper.openCamera()
            else requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        } else {
            viewBinding.viewFinder.surfaceTextureListener = textureListener
        }
        if (shouldClearPrediction) {
            viewBinding.predictedLetterTextView.text = "No Prediction Yet"
            shouldClearPrediction = false
        }
    }

    override fun onPause() {
        if (isRecording) stopProcessingAndRecording()
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
        super.onPause()
    }

    override fun onDestroy() {
        mainScope.cancel()
        videoProcessor?.close()
        videoProcessor = null
        super.onDestroy()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }
}
