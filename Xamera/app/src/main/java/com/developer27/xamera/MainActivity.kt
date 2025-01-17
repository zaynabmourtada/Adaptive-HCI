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
 * - Shows real-time processed frames (not saved) while recording raw video.
 * - Only raw video is saved when you stop tracking.
 * - Once you stop tracking, the screen (processed overlay) is cleared,
 *   but is shown again whenever you start tracking the next time.
 */
class MainActivity : AppCompatActivity() {

    // ------------------------------------------------------------------------------------
    // Fields
    // ------------------------------------------------------------------------------------
    private lateinit var viewBinding: ActivityMainBinding
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var cameraManager: CameraManager
    private lateinit var tempRecorderHelper: TempRecorderHelper
    private lateinit var cameraHelper: CameraHelper

    // Real-time video processing
    private lateinit var videoProcessor: VideoProcessor
    private var isProcessingFrame = false

    private var isTracking = false // "tracking" state = recording + processing
    private var isFrontCamera = false

    // Optional PyTorch module
    private var bestModule: Module? = null

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

    // TextureView callback
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
            // If user toggled on "tracking," do real-time processing each frame
            if (isTracking) {
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

        // Initially show the processedFrameView (we can hide it after stopTracking)
        viewBinding.processedFrameView.visibility = View.VISIBLE

        // SharedPreferences
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)

        // Initialize helpers
        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        tempRecorderHelper = TempRecorderHelper(this, cameraHelper, sharedPreferences, viewBinding)

        // VideoProcessor
        videoProcessor = VideoProcessor(this)
        // If you have a PyTorch model, you can load it:
        // videoProcessor.setModel(someModule)

        // Listen for preference changes (if you have advanced settings)
        sharedPreferences.registerOnSharedPreferenceChangeListener { _, key ->
            if (key == "shutter_speed") {
                cameraHelper.updateShutterSpeed()
            }
        }

        // Permissions
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

        // Check permissions at startup
        if (allPermissionsGranted()) {
            if (viewBinding.viewFinder.isAvailable) {
                cameraHelper.openCamera()
            } else {
                viewBinding.viewFinder.surfaceTextureListener = textureListener
            }
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        // Buttons
        // 1) Start/Stop Tracking
        viewBinding.startTrackingButton.text = "Start Tracking"
        viewBinding.startTrackingButton.setOnClickListener {
            if (isTracking) stopTracking() else startTracking()
        }

        // 2) Switch camera
        viewBinding.switchCameraButton.setOnClickListener {
            switchCamera()
        }

        // 3) Zoom controls
        cameraHelper.setupZoomControls()

        // 4) AR Button (if you have ARActivity)
        viewBinding.arButton.setOnClickListener {
            startActivity(Intent(this, ARActivity::class.java))
        }

        // 5) About & Settings
        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivityForResult(Intent(this, SettingsActivity::class.java), SETTINGS_REQUEST_CODE)
        }

        // If you want to load a PyTorch model at startup
        loadBestModelOnStartupThreaded("best_optimized.torchscript")
    }

    // ------------------------------------------------------------------------------------
    // Real-Time Processing
    // ------------------------------------------------------------------------------------
    private fun processFrameWithVideoProcessor() {
        // Avoid overlapping frames
        if (isProcessingFrame) return
        val currentBitmap = viewBinding.viewFinder.bitmap ?: return

        isProcessingFrame = true
        // Potentially do this on a background thread
        val processedBitmap = videoProcessor.processFrame(currentBitmap)
        if (processedBitmap != null) {
            // Show the processed frame
            viewBinding.processedFrameView.setImageBitmap(processedBitmap)
        }
        isProcessingFrame = false
    }

    // ------------------------------------------------------------------------------------
    // Start/Stop Tracking
    // ------------------------------------------------------------------------------------
    private fun startTracking() {
        isTracking = true
        viewBinding.startTrackingButton.text = "Stop Tracking"
        viewBinding.startTrackingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)

        // If we previously hid the processedFrameView, show it again
        viewBinding.processedFrameView.visibility = View.VISIBLE

        // Clear old data in the processor
        videoProcessor.clearTrackingData()

        // Begin recording the raw video (will be saved when stopped)
        tempRecorderHelper.startRecordingVideo()

        Toast.makeText(this, "Tracking started (raw + processed).", Toast.LENGTH_SHORT).show()
    }

    private fun stopTracking() {
        isTracking = false
        viewBinding.startTrackingButton.text = "Start Tracking"
        viewBinding.startTrackingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.blue)

        // Stop recording => raw video is saved
        tempRecorderHelper.stopRecordingVideo()

        // Clear the processed frame and hide it
        viewBinding.processedFrameView.setImageBitmap(null)
        viewBinding.processedFrameView.visibility = View.GONE

        Toast.makeText(this, "Tracking stopped. Raw video saved. Overlay cleared.", Toast.LENGTH_SHORT).show()
    }

    // ------------------------------------------------------------------------------------
    // Optional: Load PyTorch model on startup
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

    /** Copies a TorchScript model asset to internal storage, returning the file path. */
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
        if (isTracking) stopTracking()
        cameraHelper.closeCamera()
        cameraHelper.stopBackgroundThread()
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
    // Switch camera
    // ------------------------------------------------------------------------------------
    private fun switchCamera() {
        if (isTracking) stopTracking()
        isFrontCamera = !isFrontCamera
        cameraHelper.isFrontCamera = isFrontCamera
        cameraHelper.closeCamera()
        cameraHelper.openCamera()
    }
}
