plugins {
    // Plugin for Android application
    id("com.android.application")

    // Plugin for Kotlin Android integration
    id("org.jetbrains.kotlin.android")
}

android {
    // The namespace used for the app's package
    namespace = "com.developer27.xamera"

    // The SDK version the app will compile against
    compileSdk = 34

    defaultConfig {
        // Unique application identifier
        applicationId = "com.developer27.xamera"

        // Minimum supported Android SDK version
        minSdk = 26

        // Targeted Android SDK version
        targetSdk = 34

        // Version code and version name for the app (for versioning)
        versionCode = 1
        versionName = "1.0"

        // Instrumentation test runner for unit tests
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        // Configuration for release build
        release {
            // Disables code shrinking for the release version
            isMinifyEnabled = false

            // Proguard configuration files for code optimization and obfuscation (optional)
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        // Specifies Java compatibility version for compiling source code
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    buildFeatures {
        // Enables view binding to bind XML views directly in the code
        viewBinding = true
    }

    kotlinOptions {
        // Specifies the target version of the Java Virtual Machine for Kotlin
        jvmTarget = "1.8"
    }
}

dependencies {
    // Preference library for settings
    implementation("androidx.preference:preference-ktx:1.2.1")

    // CameraX library versions for camera functionality
    //-----------------------------------------------------------
    var camerax_version = "1.2.2"

    // Core dependencies for CameraX
    implementation("androidx.camera:camera-core:${camerax_version}")

    // Camera2 API support with CameraX
    implementation("androidx.camera:camera-camera2:${camerax_version}")

    // Lifecycle-aware components for CameraX
    implementation("androidx.camera:camera-lifecycle:${camerax_version}")

    // Video capturing support with CameraX
    implementation("androidx.camera:camera-video:${camerax_version}")

    // Provides a CameraX view class for preview and interactions
    implementation("androidx.camera:camera-view:${camerax_version}")

    // Extensions to enhance camera functionality like HDR, night mode, etc.
    implementation("androidx.camera:camera-extensions:${camerax_version}")

    // Android core libraries for Kotlin extensions
    implementation("androidx.core:core-ktx:1.13.1")

    // Support for backward-compatible Android components
    implementation("androidx.appcompat:appcompat:1.6.1")

    // Google Material Design library
    implementation("com.google.android.material:material:1.11.0")

    // ConstraintLayout for flexible UI designs
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // JUnit testing library for unit testing
    testImplementation("junit:junit:4.13.2")

    // Android extension for JUnit
    androidTestImplementation("androidx.test.ext:junit:1.2.1")

    // Espresso testing library for UI testing
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}