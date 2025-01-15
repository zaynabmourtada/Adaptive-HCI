package com.developer27.xamera
// If you need coroutines or advanced logic:
// import androidx.lifecycle.lifecycleScope
// import kotlinx.coroutines.Dispatchers
// import kotlinx.coroutines.launch
// import kotlinx.coroutines.withContext

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
    private lateinit var tempRecorderHelper: TempRecorderHelper
    private lateinit var cameraHelper: CameraHelper

    private var isTracking = false
    private var isFrontCamera = false

    // If you eventually want to load a single best.pt model, you could do something like:
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

    // ------------------------------------------------------------------------------------
    // VideoProcessor references from older code (commented out)
    // (One day, you'll replace TempRecorderHelper with real VideoProcessor.)
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
            // if we had real-time processing: processFrameWithVideoProcessor()
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

        // Initialize the helpers
        cameraHelper = CameraHelper(this, viewBinding, sharedPreferences)
        tempRecorderHelper = TempRecorderHelper(this, cameraHelper, sharedPreferences, viewBinding)

        // Observe shutter_speed changes
        sharedPreferences.registerOnSharedPreferenceChangeListener { prefs, key ->
            if (key == "shutter_speed") {
                cameraHelper.updateShutterSpeed()  // update preview shutter speed
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

        // Buttons
        viewBinding.startTrackingButton.setOnClickListener {
            if (isTracking) stopTracking() else startTracking()
        }
        viewBinding.switchCameraButton.setOnClickListener {
            switchCamera()
        }

        cameraHelper.setupZoomControls()

        viewBinding.aboutButton.setOnClickListener {
            startActivity(Intent(this, AboutXameraActivity::class.java))
        }
        viewBinding.settingsButton.setOnClickListener {
            startActivityForResult(Intent(this, SettingsActivity::class.java), SETTINGS_REQUEST_CODE)
        }

        // If we wanted to load PyTorch models at startup
        loadBestModelOnStartupThreaded("YOLOv2-Mobile.torchscript")
    }

    private fun loadBestModelOnStartupThreaded(bestModel: String) {
        // 1) Launch a plain background Thread
        Thread {
            // 2) Copy asset => load model
            val bestLoadedPath = copyAssetModelBlocking(bestModel)

            // 3) Switch to main thread to update UI
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
     * A blocking function to copy best.pt from assets to internal storage.
     * This does NOT use coroutines.
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
    // Start/Stop Tracking => Start/Stop Recording
    // ------------------------------------------------------------------------------------
    private fun startTracking() {
        isTracking = true
        viewBinding.startTrackingButton.text = "Stop Tracking"
        viewBinding.startTrackingButton.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.red)

        // If we had a videoProcessor:
        // videoProcessor.clearTrackingData()
        // viewBinding.processedFrameView.visibility = View.VISIBLE

        tempRecorderHelper.startRecordingVideo() // ***Temporary**: will be replaced by VideoProcessor
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

        tempRecorderHelper.stopRecordingVideo() // ***Temporary**: will be replaced by VideoProcessor
        Toast.makeText(this, "Recording stopped.", Toast.LENGTH_SHORT).show()
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
