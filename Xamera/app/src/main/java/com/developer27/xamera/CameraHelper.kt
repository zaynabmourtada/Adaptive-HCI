package com.developer27.xamera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.SharedPreferences
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
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.MotionEvent
import android.view.Surface
import android.widget.Toast
import androidx.annotation.RequiresPermission
import com.developer27.xamera.databinding.ActivityMainBinding

/**
 * CameraHelper is responsible for:
 *  - Opening & closing the camera
 *  - Switching front/back
 *  - Creating a preview
 *  - Handling zoom & shutter speed
 *  - Starting a background thread for camera operations
 *
 * The actual recording logic is in TempRecorderHelper.
 */
class CameraHelper(
    private val activity: MainActivity,
    private val viewBinding: ActivityMainBinding,
    private val sharedPreferences: SharedPreferences
) {
    // The Android Camera2 API
    private val cameraManager: CameraManager by lazy {
        activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    var cameraDevice: CameraDevice? = null
    var cameraCaptureSession: CameraCaptureSession? = null
    var captureRequestBuilder: CaptureRequest.Builder? = null

    var previewSize: Size? = null
    var videoSize: Size? = null
    var sensorArraySize: Rect? = null

    // Switch between front/back camera
    var isFrontCamera = false

    // Background thread + handler
    private var backgroundThread: HandlerThread? = null
    var backgroundHandler: Handler? = null
        private set

    // Zoom
    private var zoomLevel = 1.0f
    private val maxZoom = 10.0f

    /**
     * StateCallback for the camera device
     */
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
            activity.finish()
        }
    }

    // ------------------------------------------------------------------------------------
    // Background thread management
    // ------------------------------------------------------------------------------------
    fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    fun stopBackgroundThread() {
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
    // Camera open/close
    // ------------------------------------------------------------------------------------
    @SuppressLint("MissingPermission")
    @RequiresPermission(Manifest.permission.CAMERA)
    fun openCamera() {
        try {
            val cameraId = getCameraId()
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            sensorArraySize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)

            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                ?: return

            previewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture::class.java))
            videoSize = chooseOptimalSize(map.getOutputSizes(MediaRecorder::class.java))

            cameraManager.openCamera(cameraId, stateCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        } catch (e: SecurityException) {
            e.printStackTrace()
            Toast.makeText(activity, "Camera permission needed.", Toast.LENGTH_SHORT).show()
        }
    }

    fun closeCamera() {
        cameraCaptureSession?.close()
        cameraCaptureSession = null
        cameraDevice?.close()
        cameraDevice = null
    }

    // ------------------------------------------------------------------------------------
    // Create preview
    // ------------------------------------------------------------------------------------
    fun createCameraPreview() {
        try {
            val texture = viewBinding.viewFinder.surfaceTexture ?: return
            previewSize?.let { texture.setDefaultBufferSize(it.width, it.height) }

            val surface = Surface(texture)
            captureRequestBuilder = cameraDevice?.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            captureRequestBuilder?.addTarget(surface)

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
                        cameraCaptureSession = session
                        updatePreview()
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Toast.makeText(
                            activity,
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

    fun updatePreview() {
        if (cameraDevice == null) return
        try {
            captureRequestBuilder?.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
            applyRollingShutter()
            updateShutterSpeed()
            applyFlashIfEnabled()
            applyLightingMode()

            cameraCaptureSession?.setRepeatingRequest(
                captureRequestBuilder!!.build(),
                null,
                backgroundHandler
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    // ------------------------------------------------------------------------------------
    // getCameraId to pick front/back
    // ------------------------------------------------------------------------------------
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

    private fun chooseOptimalSize(choices: Array<Size>): Size {
        val targetWidth = 1280
        val targetHeight = 720
        val found720p = choices.find { it.width == targetWidth && it.height == targetHeight }
        if (found720p != null) {
            return found720p
        }
        // fallback: pick smallest
        return choices.minByOrNull { it.width * it.height } ?: choices[0]
    }

    // ------------------------------------------------------------------------------------
    // Rolling shutter, shutter speed, lighting, flash
    // ------------------------------------------------------------------------------------
    fun applyRollingShutter() {
        val shutterSetting = sharedPreferences.getString("shutter_speed", "15")?.toInt() ?: 15
        val shutterValue = if (shutterSetting >= 5) 1000000000L / shutterSetting else 0L
        captureRequestBuilder?.apply {
            if (shutterValue > 0) {
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
                set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterValue)
            } else {
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            }
        }
    }

    fun updateShutterSpeed() {
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
                cameraCaptureSession?.setRepeatingRequest(build(), null, backgroundHandler)
            } catch (e: CameraAccessException) {
                e.printStackTrace()
            }
        }
    }

    fun applyFlashIfEnabled() {
        val isFlashEnabled = sharedPreferences.getBoolean("enable_flash", false)
        captureRequestBuilder?.set(
            CaptureRequest.FLASH_MODE,
            if (isFlashEnabled) CaptureRequest.FLASH_MODE_TORCH
            else CaptureRequest.FLASH_MODE_OFF
        )
    }

    fun applyLightingMode() {
        val lightingMode = sharedPreferences.getString("lighting_mode", "normal")
        val compensationRange = cameraManager.getCameraCharacteristics(getCameraId())
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

    // ------------------------------------------------------------------------------------
    // Zoom
    // ------------------------------------------------------------------------------------
    fun setupZoomControls() {
        val zoomHandler = Handler(activity.mainLooper)
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

    fun applyZoom() {
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
        cameraCaptureSession?.setRepeatingRequest(
            captureRequestBuilder!!.build(),
            null,
            backgroundHandler
        )
    }
}
