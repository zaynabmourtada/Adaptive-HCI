package com.developer27.xamera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.os.Bundle
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
import com.developer27.xamera.camera.TempRecorderHelper
import com.developer27.xamera.databinding.ActivityMainBinding
import com.developer27.xamera.openGL2D.OpenGL2DActivity
import com.developer27.xamera.openGL3D.OpenGL3DActivity
import com.developer27.xamera.videoprocessing.VideoProcessor
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * MainActivity for the Xamera app:
 * - Starts/stops real-time processing (VideoProcessor) & raw video recording (TempRecorderHelper).
 *
 * Flow:
 *   [Start Processing] => Start video + real-time processing overlay
 *   [Stop Processing]  => Stop video + real-time processing overlay
 *   Then user can click "3D" to see final path in OpenGL.
 */
class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper

    private lateinit var tempRecorderHelper: TempRecorderHelper

    // Real-time
    private var isRecording = false
    private var isProcessing = false

    private var videoProcessor: VideoProcessor? = null
    private var tfliteInterpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    private var isProcessingFrame = false

    // We'll store final line coords here after "Stop Processing".
    // Each element is [x, y, z] in OpenGL coordinate space.
    private var finalLineCoords = mutableListOf<List<Float>>()

    // Suppose your camera frames are 640×480
    private val frameWidth = 640f
    private val frameHeight = 480f

    private val REQUIRED_PERMISSIONS = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<Array<String>>

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

        try {
            System.loadLibrary("tensorflowlite_gpu_jni")
            Log.d("TFLiteLoader", "✅ Successfully loaded GPU JNI library")
        } catch (e: UnsatisfiedLinkError) {
            Log.e("TFLiteLoader", "❌ Failed to load GPU JNI library. Falling back to CPU.", e)
        }

        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        viewBinding.processedFrameView.visibility = View.GONE

        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)

        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        tempRecorderHelper = TempRecorderHelper(this, cameraHelper, sharedPreferences, viewBinding)

        // Listen for shutter_speed changes
        sharedPreferences.registerOnSharedPreferenceChangeListener { _, key ->
            if (key == "shutter_speed") {
                cameraHelper.updateShutterSpeed()
            }
        }

        requestPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            val camGranted = permissions[Manifest.permission.CAMERA] ?: false
            val micGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
            if (camGranted && micGranted) {
                if (viewBinding.viewFinder.isAvailable) {
                    cameraHelper.openCamera()
                } else {
                    viewBinding.viewFinder.surfaceTextureListener = textureListener
                }
            } else {
                Toast.makeText(
                    this,
                    "Camera & Audio permissions are required.",
                    Toast.LENGTH_SHORT
                ).show()
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

        videoProcessor = VideoProcessor(this)

        // Start/Stop
        viewBinding.startProcessingButton.setOnClickListener {
            if (isRecording) {
                stopProcessingAndRecording()
            } else {
                startProcessingAndRecording()
            }
        }
        // Switch cam
        viewBinding.switchCameraButton.setOnClickListener {
            switchCamera()
        }
        // Zoom
        cameraHelper.setupZoomControls()

        // 2D
        viewBinding.twoDOnlyButton.setOnClickListener {
            launch2DOnlyFeature()
        }

        // 3D => pass finalLineCoords as float array
        viewBinding.threeDOnlyButton.setOnClickListener {
            launch3DOnlyFeature()
        }

        // About/Settings
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        loadTFLiteModelOnStartupThreaded("YOLOv3_float32.tflite")

        viewBinding.unityButton.setOnClickListener {
            startActivity(Intent(this, com.unity3d.player.UnityPlayerGameActivity::class.java))
        }
    }

    // This function triggers 2D OpenGL feature
    private fun launch2DOnlyFeature() {
        try {
            val intent = Intent(this, OpenGL2DActivity::class.java)
            startActivity(intent)
        } catch (e: Exception) {
            Toast.makeText(this, "Error launching 2D feature: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    // This function triggers 3D OpenGL feature
    private fun launch3DOnlyFeature() {
        try {
            val intent = Intent(this, OpenGL3DActivity::class.java)

            val flatArray = finalLineCoords.flatten().toFloatArray()
            intent.putExtra("3D_POINTS", flatArray)
            startActivity(intent)
        } catch (e: Exception) {
            Toast.makeText(this, "Error launching 3D feature: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startProcessingAndRecording() {
        isRecording = true
        isProcessing = true
        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)

        // --- Commenting out video processing / overlay ---
        viewBinding.processedFrameView.visibility = View.VISIBLE
        videoProcessor?.clearTrackingData()

        // Commenting out - We keep only the raw video recording:
        //tempRecorderHelper.startRecordingVideo()

        Toast.makeText(this, "VideoProc started.", Toast.LENGTH_SHORT).show()
    }

    private fun stopProcessingAndRecording() {
        isRecording = false
        isProcessing = false
        viewBinding.startProcessingButton.text = "Start Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)

        // --- Commenting out video processing / overlay ---
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)

        // Commenting out - Stop video recording:
        //tempRecorderHelper.stopRecordingVideo()

        // --- Commenting out retrieval of final tracked coords ---
        // finalLineCoords.clear()
        // recieveXYZfromVideoProcessor()

        Toast.makeText(this, "VideoProc Stopped.", Toast.LENGTH_SHORT).show()
    }

    private fun recieveXYZfromVideoProcessor(){
        // TODO <Soham Naik> X, Y and Z coordinates must be transmitted from YOLO model.
        val postFilterData = videoProcessor?.getPostFilterData()
        if (postFilterData != null) {
            for (f in postFilterData) {
                // Suppose (f.x, f.y) in [0..640]x[0..480]
                // Move origin to center => x - 320, y - 240 => then scale to [-1..1]
                val centerX = f.x.toFloat() - (frameWidth / 2f)
                val centerY = f.y.toFloat() - (frameHeight / 2f)

                // Scale so that -320..320 => -1..1 in X, -240..240 => -1..1 in Y
                val ndcX = centerX / (frameWidth / 2f)    // => [-1..1]
                // Typically, we invert Y so top is +1 => so maybe:
                val ndcY = -centerY / (frameHeight / 2f)  // => [-1..1]
                val z    = 0f

                finalLineCoords.add(listOf(ndcX, ndcY, z))
            }
        }
    }

    private fun processFrameWithVideoProcessor() {
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true

        // Use the new callback-based processFrame method
        videoProcessor?.processFrame(bitmap) { processedBitmap ->
            runOnUiThread {
                if (processedBitmap != null && isProcessing) {
                    viewBinding.processedFrameView.setImageBitmap(processedBitmap)
                }
                isProcessingFrame = false
            }
        }
    }

    private fun loadTFLiteModelOnStartupThreaded(modelName: String) {
        Thread {
            val modelPath = copyTFLiteModelBlocking(modelName) // Copy model from assets
            runOnUiThread {
                if (modelPath.isNotEmpty()) {
                    try {
                        // Correct instantiation of TFLite Interpreter
                        tfliteInterpreter = Interpreter(loadModelFile(modelPath))

                        // Pass the model to VideoProcessor
                        videoProcessor?.setTFLiteModel(tfliteInterpreter!!)

                        Toast.makeText(
                            this@MainActivity,
                            "TFLite Model loaded: $modelName",
                            Toast.LENGTH_SHORT
                        ).show()
                    } catch (e: Exception) {
                        Toast.makeText(
                            this@MainActivity,
                            "Error loading model: ${e.message}",
                            Toast.LENGTH_LONG
                        ).show()
                        Log.e("MainActivity", "TFLite Interpreter.load() error", e)
                    }
                } else {
                    Toast.makeText(
                        this@MainActivity,
                        "Failed to copy or load $modelName",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }.start()
    }

    /** Cleanup Function: Call This When Activity is Destroyed */
    private fun releaseTFLiteResources() {
        try {
            tfliteInterpreter?.close()
            gpuDelegate?.close() // ✅ Properly release the GPU delegate
            Log.d("TFLiteLoader", "FLite Interpreter & GPU Delegate released!")
        } catch (e: Exception) {
            Log.e("TFLiteLoader", "Error releasing TFLite resources: ${e.message}", e)
        }
    }

    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val file = File(modelPath)
        val inputStream = FileInputStream(file)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    private fun copyTFLiteModelBlocking(assetName: String): String {
        return try {
            val outFile = File(filesDir, assetName)

            // If file already exists and is valid, return its path
            if (outFile.exists() && outFile.length() > 0) {
                Log.d("TFLiteLoader", "Model already exists: ${outFile.absolutePath}")
                return outFile.absolutePath
            }

            // Copy from assets to internal storage
            assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }

            Log.d("TFLiteLoader", "Successfully copied model to: ${outFile.absolutePath}")
            outFile.absolutePath
        } catch (e: Exception) {
            Log.e("TFLiteLoader", "Error copying TFLite model $assetName: ${e.message}", e)
            ""
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == SETTINGS_REQUEST_CODE && resultCode == RESULT_OK) {
            cameraHelper.updateShutterSpeed()
            Toast.makeText(this, "Settings updated", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onResume() {
        super.onResume()
        cameraHelper.startBackgroundThread()
        if (viewBinding.viewFinder.isAvailable) {
            if (allPermissionsGranted()) {
                cameraHelper.openCamera()
                cameraHelper.updateShutterSpeed()
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

    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
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
}