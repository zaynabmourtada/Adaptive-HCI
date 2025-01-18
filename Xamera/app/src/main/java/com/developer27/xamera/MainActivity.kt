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
import com.developer27.xamera.databinding.ActivityMainBinding
import org.pytorch.Module
import java.io.File
import java.io.FileOutputStream

/**
 * MainActivity for the Xamera app:
 * - Starts and stops real-time processing (VideoProcessor) AND
 *   simultaneously records raw video (TempRecorderHelper).
 *
 * Flow:
 *   [Start Processing] => (1) Start raw video recording
 *                        (2) Start real-time processing overlay
 *   [Stop Processing]  => (1) Stop raw video recording & save file
 *                        (2) Stop real-time processing overlay
 */
class MainActivity : AppCompatActivity() {

    // ------------------------------------------------------------------------------------
    // Fields
    // ------------------------------------------------------------------------------------
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var cameraHelper: CameraHelper

    private lateinit var tempRecorderHelper: TempRecorderHelper
    private var bestModule: Module? = null

    // Real-time processing state
    private var isProcessing = false
    private var videoProcessor: VideoProcessor? = null
    private var isProcessingFrame = false

    // Required permissions
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

    // TextureView callback: open camera when surface is ready
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

        // Called every time there’s a new preview frame
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
            if (isProcessing) {
                processFrameWithVideoProcessor()
            }
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

        // Hide processedFrameView if not needed
        viewBinding.processedFrameView.visibility = View.GONE

        // SharedPreferences
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)

        // Initialize helpers
        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        tempRecorderHelper = TempRecorderHelper(this, cameraHelper, sharedPreferences, viewBinding)

        // If the user changes shutter_speed in Settings
        sharedPreferences.registerOnSharedPreferenceChangeListener { _, key ->
            if (key == "shutter_speed") {
                cameraHelper.updateShutterSpeed()
            }
        }

        // Permission launcher
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

        // If permissions are already granted
        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) {
                cameraHelper.openCamera()
            } else {
                viewBinding.viewFinder.surfaceTextureListener = textureListener
            }
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        // Initialize the advanced VideoProcessor if desired
        videoProcessor = VideoProcessor(this)

        // Buttons
        // 1) Start or Stop Processing & Recording
        viewBinding.startProcessingButton.setOnClickListener {
            if (isProcessing) {
                stopProcessingAndRecording()
            } else {
                startProcessingAndRecording()
            }
        }

        // 2) Switch camera
        viewBinding.switchCameraButton.setOnClickListener {
            switchCamera()
        }

        // Zoom controls
        cameraHelper.setupZoomControls()

        // Listener for the "2D Only" button
        viewBinding.threeDOnlyButton.setOnClickListener {
            launch2DOnlyFeature()
        }

        // About / Settings
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }

        //Launch 3D feature when clicked
        viewBinding.settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        // If we wanted to load PyTorch models at startup
        loadBestModelOnStartupThreaded("YOLOv2-Mobile.torchscript")
    }

    // launch3DOnlyFeature is responsible for creating the 2D Environment
    private fun launch2DOnlyFeature() {
        try {
            // Start UnityPlayerGameActivity or any Unity-based activity
            val intent = Intent(this, OpenGL2DActivity::class.java)
            startActivity(intent)
        } catch (e: Exception) {
            Toast.makeText(this, "Error launching 3D feature: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    // ------------------------------------------------------------------------------------
    // Start Processing & Recording
    // ------------------------------------------------------------------------------------
    private fun startProcessingAndRecording() {
        isProcessing = true

        // Change button appearance
        viewBinding.startProcessingButton.text = "Stop Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)

        // Show processed-frame overlay
        viewBinding.processedFrameView.visibility = View.VISIBLE

        // Clear old data in VideoProcessor
        videoProcessor?.clearTrackingData()

        // Start the raw video recorder => saving unprocessed video file
        tempRecorderHelper.startRecordingVideo()

        Toast.makeText(this, "Processing + Recording started.", Toast.LENGTH_SHORT).show()
    }

    // ------------------------------------------------------------------------------------
    // Stop Processing & Recording
    // ------------------------------------------------------------------------------------
    private fun stopProcessingAndRecording() {
        isProcessing = false

        // Change button appearance
        viewBinding.startProcessingButton.text = "Start Tracking"
        viewBinding.startProcessingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)

        // Hide the processed-frame overlay
        viewBinding.processedFrameView.visibility = View.GONE
        viewBinding.processedFrameView.setImageBitmap(null)

        // Stop the raw video recorder => finalize/save video file
        tempRecorderHelper.stopRecordingVideo()

        Toast.makeText(this, "Processing + Recording stopped.", Toast.LENGTH_SHORT).show()
    }

    // ------------------------------------------------------------------------------------
    // Real-time frame processing
    // ------------------------------------------------------------------------------------
    private fun processFrameWithVideoProcessor() {
        // Avoid re-entrancy
        if (isProcessingFrame) return
        val bitmap = viewBinding.viewFinder.bitmap ?: return
        isProcessingFrame = true

        // Run on a background thread
        Thread {
            val processedBitmap = videoProcessor?.processFrame(bitmap)
            runOnUiThread {
                if (processedBitmap != null && isProcessing) {
                    viewBinding.processedFrameView.setImageBitmap(processedBitmap)
                }
                isProcessingFrame = false
            }
        }.start()
    }

    // ------------------------------------------------------------------------------------
    // Model Loading (optional)
    // ------------------------------------------------------------------------------------
    private fun loadBestModelOnStartupThreaded(bestModel: String) {
        Thread {
            val bestLoadedPath = copyAssetModelBlocking(bestModel)
            runOnUiThread {
                if (bestLoadedPath.isNotEmpty()) {
                    try {
                        bestModule = Module.load(bestLoadedPath)
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

    /**
     * Copy model asset to internal storage (blocking call)
     */
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

    // ------------------------------------------------------------------------------------
    // onActivityResult
    // ------------------------------------------------------------------------------------
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == SETTINGS_REQUEST_CODE && resultCode == RESULT_OK) {
            cameraHelper.updateShutterSpeed()
            Toast.makeText(this, "Settings updated", Toast.LENGTH_SHORT).show()
        }
    }

    // ------------------------------------------------------------------------------------
    // onResume / onPause
    // ------------------------------------------------------------------------------------
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
        // If currently processing, stop
        if (isProcessing) {
            stopProcessingAndRecording()
        }
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
        super.onPause()
    }

    // ------------------------------------------------------------------------------------
    // Permissions
    // ------------------------------------------------------------------------------------
    private fun allPermissionsGranted(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    // ------------------------------------------------------------------------------------
    // Switch camera
    // ------------------------------------------------------------------------------------
    private var isFrontCamera = false
    private fun switchCamera() {
        // If we’re processing, stop first
        if (isProcessing) {
            stopProcessingAndRecording()
        }
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        cameraHelper.openCamera()
    }
}