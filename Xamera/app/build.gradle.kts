plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.developer27.xamera"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.developer27.xamera"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86_64")
        }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    packaging {
        jniLibs {
            pickFirsts.add("lib/x86/libc++_shared.so")
            pickFirsts.add("lib/x86_64/libc++_shared.so")
            pickFirsts.add("lib/armeabi-v7a/libc++_shared.so")
            pickFirsts.add("lib/arm64-v8a/libc++_shared.so")
            pickFirsts.add("lib/arm64-v8a/libtensorflowlite_gpu_jni.so")
            pickFirsts.add("lib/armeabi-v7a/libtensorflowlite_gpu_jni.so")
        }
    }
    androidResources {
        noCompress += listOf("tflite")
    }
}

dependencies {
    // OpenCV
    implementation(project(":OpenCV-4.10.0")) {
        exclude(group = "org.bytedeco", module = "libc++_shared")
    }

    //implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.aar"))))
    implementation(files("libs/ar-app-release.aar"))

    // ML Kit, etc.
    implementation("com.google.mlkit:vision-common:17.3.0")

    // TensorFlow Lite (For GPU Utilization)
    implementation("com.google.ai.edge.litert:litert:1.1.0") // Core TFLite runtime
    implementation("com.google.ai.edge.litert:litert-gpu:1.1.0") // GPU acceleration
    implementation("com.google.ai.edge.litert:litert-support:1.1.0") // Support library

    // CameraX
    val cameraxVersion = "1.2.2"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-video:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")
    implementation("androidx.camera:camera-extensions:$cameraxVersion")

    // ARCore
    implementation("com.google.ar:core:1.46.0")

    // Kotlin & Android core libs
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // Preferences
    implementation("androidx.preference:preference-ktx:1.2.1")

    //Splash screen
    implementation("androidx.core:core-splashscreen:1.0.0")

    // Testing
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")

    // Apache Commons Math
    implementation("org.apache.commons:commons-math3:3.6.1")

}
