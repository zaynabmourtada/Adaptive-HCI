package com.developer27.xamera

import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.media.MediaRecorder
import android.os.Environment
import android.util.Log
import android.util.Size
import android.view.Surface
import android.widget.Toast
import com.developer27.xamera.databinding.ActivityMainBinding
import java.io.File

/**
 * TempRecorderHelper is a *temporary* class for MediaRecorder logic.
 * In the future, you'll replace this with your actual "VideoProcessor".
 */
class TempRecorderHelper(
    private val mainActivity: MainActivity,
    private val cameraHelper: CameraHelper,
    private val sharedPreferences: android.content.SharedPreferences,
    private val viewBinding: ActivityMainBinding
) {
    private var mediaRecorder: MediaRecorder? = null
    private var outputFile: File? = null

    /**
     * Start recording video using rolling shutter if the camera is ready.
     */
    fun startRecordingVideo() {
        if (cameraHelper.cameraDevice == null) {
            Toast.makeText(
                mainActivity,
                "CameraDevice not ready.",
                Toast.LENGTH_SHORT
            ).show()
            return
        }

        // Release old recorder if needed
        if (mediaRecorder != null) {
            try { mediaRecorder?.stop() } catch (_: Exception) {}
            mediaRecorder?.reset()
            mediaRecorder?.release()
            mediaRecorder = null
        }

        try {
            mediaRecorder = MediaRecorder().apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setVideoSource(MediaRecorder.VideoSource.SURFACE)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            }

            // Public Movies folder
            @Suppress("DEPRECATION")
            val moviesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES)
            if (moviesDir == null || (!moviesDir.exists() && !moviesDir.mkdirs())) {
                Toast.makeText(
                    mainActivity,
                    "Cannot access public Movies folder.",
                    Toast.LENGTH_LONG
                ).show()
                return
            }

            outputFile = File(moviesDir, "Xamera_${System.currentTimeMillis()}.mp4")
            mediaRecorder?.setOutputFile(outputFile!!.absolutePath)

            // For example, 1280x720:
            val recordSize = cameraHelper.videoSize ?: Size(1280, 720)
            mediaRecorder?.apply {
                setVideoEncoder(MediaRecorder.VideoEncoder.H264)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                setVideoEncodingBitRate(2_000_000)
                setVideoFrameRate(30)
                setVideoSize(recordSize.width, recordSize.height)
                prepare()
            }

            val texture = viewBinding.viewFinder.surfaceTexture ?: return
            cameraHelper.previewSize?.let { texture.setDefaultBufferSize(it.width, it.height) }

            val previewSurface = Surface(texture)
            val recorderSurface = mediaRecorder!!.surface

            cameraHelper.captureRequestBuilder =
                cameraHelper.cameraDevice?.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)

            cameraHelper.captureRequestBuilder?.addTarget(previewSurface)
            cameraHelper.captureRequestBuilder?.addTarget(recorderSurface)

            // Rolling shutter for RECORD
            applyRollingShutterForRecording(cameraHelper.captureRequestBuilder)

            cameraHelper.cameraDevice?.createCaptureSession(
                listOf(previewSurface, recorderSurface),
                object : android.hardware.camera2.CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: android.hardware.camera2.CameraCaptureSession) {
                        cameraHelper.cameraCaptureSession = session
                        try {
                            cameraHelper.captureRequestBuilder?.set(
                                CaptureRequest.CONTROL_MODE,
                                CameraMetadata.CONTROL_MODE_OFF
                            )
                            cameraHelper.cameraCaptureSession?.setRepeatingRequest(
                                cameraHelper.captureRequestBuilder!!.build(),
                                null,
                                cameraHelper.backgroundHandler
                            )
                            mediaRecorder?.start()
                        } catch (e: android.hardware.camera2.CameraAccessException) {
                            Toast.makeText(
                                mainActivity,
                                "Failed to start recording: ${e.message}",
                                Toast.LENGTH_LONG
                            ).show()
                            e.printStackTrace()
                        }
                    }

                    override fun onConfigureFailed(session: android.hardware.camera2.CameraCaptureSession) {
                        Toast.makeText(
                            mainActivity,
                            "Capture session config failed.",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                },
                cameraHelper.backgroundHandler
            )

        } catch (e: Exception) {
            Log.e("TempRecorderHelper", "MediaRecorder error: ${e.message}", e)
            Toast.makeText(
                mainActivity,
                "Cannot record: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
            mediaRecorder?.reset()
            mediaRecorder?.release()
            mediaRecorder = null
        }
    }

    /**
     * Apply rolling shutter specifically for the RECORD request.
     */
    private fun applyRollingShutterForRecording(builder: CaptureRequest.Builder?) {
        if (builder == null) return
        val shutterSetting = sharedPreferences.getString("shutter_speed", "15")?.toInt() ?: 15
        val shutterValue = if (shutterSetting >= 5) 1000000000L / shutterSetting else 0L

        if (shutterValue > 0) {
            builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
            builder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterValue)
        } else {
            builder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        }
    }

    /**
     * Stop recording video and restore camera preview.
     */
    fun stopRecordingVideo() {
        if (mediaRecorder == null) return
        try {
            mediaRecorder?.stop()
        } catch (e: Exception) {
            Log.e("TempRecorderHelper", "Error stopping recording: ${e.message}", e)
        }
        mediaRecorder?.reset()
        mediaRecorder?.release()
        mediaRecorder = null

        // Return to normal preview
        cameraHelper.createCameraPreview()

        outputFile?.let { file ->
            if (file.exists()) {
                Toast.makeText(
                    mainActivity,
                    "Video saved:\n${file.absolutePath}",
                    Toast.LENGTH_LONG
                ).show()
            } else {
                Toast.makeText(mainActivity, "No output file found.", Toast.LENGTH_SHORT).show()
            }
        }
    }
}
