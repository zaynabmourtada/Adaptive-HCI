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
import org.pytorch.Module
import java.io.File
import java.io.FileOutputStream

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
    private var bestModule: Module? = null

    // Real-time
    private var isRecording = false //Temproary
    private var isProcessing = false
    private var videoProcessor: VideoProcessor? = null
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

        // Optionally load model
        loadBestModelOnStartupThreaded("YOLOv2-Mobile.torchscript")

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

        Toast.makeText(this, "Processing + Recording started.", Toast.LENGTH_SHORT).show()
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

        Toast.makeText(this, "Processing + Recording stopped.", Toast.LENGTH_SHORT).show()
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

    private fun loadBestModelOnStartupThreaded(bestModel: String) {
        Thread {
            val bestLoadedPath = copyAssetModelBlocking(bestModel)
            runOnUiThread {
                if (bestLoadedPath.isNotEmpty()) {
                    try {
                        val bestModule = org.pytorch.Module.load(bestLoadedPath)
                        videoProcessor?.setModel(bestModule) // Pass the model to VideoProcessor
                        Toast.makeText(
                            this@MainActivity,
                            "YOLO Model loaded: $bestModel",
                            Toast.LENGTH_SHORT
                        ).show()
                    } catch (e: Exception) {
                        Toast.makeText(
                            this@MainActivity,
                            "Error loading model: ${e.message}",
                            Toast.LENGTH_LONG
                        ).show()
                        Log.e("MainActivity", "Module.load() error", e)
                    }
                } else {
                    Toast.makeText(
                        this@MainActivity,
                        "Failed to copy or load $bestModel",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }.start()
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