plugins {
    // Plugin for Android application
    id("com.android.application")

    // Plugin for Kotlin Android integration
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.developer27.xamera"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.developer27.xamera"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a")
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
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    buildFeatures {
        viewBinding = true
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    packagingOptions {
        pickFirst("lib/armeabi-v7a/libc++_shared.so")
        pickFirst("lib/arm64-v8a/libc++_shared.so")
    }
}

dependencies {
    implementation("androidx.preference:preference-ktx:1.2.1")
    implementation(project(":OpenCV-4.10.0"))
    implementation(project(":OpenCV-4.10.0"))
    implementation("com.google.mlkit:vision-common:17.3.0")

    // CameraX dependencies for camera functionality
    var camerax_version = "1.2.2"
    implementation("androidx.camera:camera-core:$camerax_version")
    implementation("androidx.camera:camera-camera2:$camerax_version")
    implementation("androidx.camera:camera-lifecycle:$camerax_version")
    implementation("androidx.camera:camera-video:$camerax_version")
    implementation("androidx.camera:camera-view:$camerax_version")
    implementation("androidx.camera:camera-extensions:$camerax_version")

    // Camera2 dependencies (comes with the Android SDK but can add if needed)
    implementation("androidx.camera:camera-camera2:${camerax_version}")

    // Add PyTorch dependencies
    implementation("org.pytorch:pytorch_android:1.13.1")
    implementation("org.pytorch:pytorch_android_torchvision:1.13.1")

    // Android and Kotlin core libraries
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // Unit testing dependencies
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")

    //Apache Commons Math
    implementation("org.apache.commons:commons-math3:3.6.1")
}
