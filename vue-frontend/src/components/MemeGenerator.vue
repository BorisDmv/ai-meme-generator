<template>
  <div class="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-blue-100 to-purple-200 p-4 animated-gradient">

    <h1 class="text-4xl font-bold mb-8 text-gray-800">AI Meme Generator</h1>

    <div
      class="relative border-4 border-dashed rounded-2xl w-full max-w-2xl h-[500px] flex items-center justify-center cursor-pointer bg-white hover:bg-blue-100 transition-all duration-300 mb-6"
      :class="{ 'border-blue-600 bg-blue-50': isDragging }"
      @dragover.prevent="onDragOver"
      @dragleave.prevent="onDragLeave"
      @drop.prevent="handleDrop"
      @click="triggerFileInput"
    >
      <transition name="fade">
        <p v-if="!selectedImage && !generatedMemeImage && !loading" class="text-gray-500 text-center px-4">
          Drag & drop an image here<br />or click to select
        </p>
      </transition>

      <transition name="fade">
        <img
          v-if="selectedImage && !generatedMemeImage && !loading"
          :src="selectedImage"
          alt="Uploaded Preview"
          class="absolute w-full h-full object-contain rounded-xl p-4"
        />
      </transition>

      <transition name="fade">
        <img
          v-if="generatedMemeImage && !loading"
          :src="generatedMemeImage"
          alt="Generated Meme"
          class="absolute w-full h-full object-contain rounded-xl p-4 shadow-md"
        />
      </transition>

      <transition name="fade">
        <div v-if="loading" class="absolute inset-0 flex flex-col items-center justify-center bg-white bg-opacity-70 rounded-xl">
          <div class="text-blue-500 font-semibold text-lg mb-2">Thinking... 🤔</div>
          <div class="loader"></div>
        </div>
      </transition>
    </div>

    <input type="file" ref="fileInput" @change="handleFileSelect" accept="image/*" class="hidden" />

    <div v-if="selectedFile && !loading" class="mt-4">
      <button
        @click="uploadImage"
        class="px-6 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-full shadow hover:from-blue-600 hover:to-purple-600 transition-all duration-300"
      >
        {{ generatedMemeImage ? 'Generate Again' : 'Generate Meme' }}
      </button>
    </div>
  </div>
</template>


<script setup>
import { ref, onUnmounted } from 'vue'
import axios from 'axios'

const selectedFile = ref(null)
const selectedImage = ref(null)
const generatedMemeImage = ref(null)
const loading = ref(false)
const fileInput = ref(null)
const isDragging = ref(false)

function handleFile(file) {
  if (file && file.type.startsWith('image/')) {
    selectedFile.value = file
    if (selectedImage.value) URL.revokeObjectURL(selectedImage.value)
    selectedImage.value = URL.createObjectURL(file)
    generatedMemeImage.value = null
  } else {
    alert('Please select a valid image file.')
    selectedFile.value = null
    selectedImage.value = null
    generatedMemeImage.value = null
  }
}

function handleDrop(e) {
  const file = e.dataTransfer.files[0]
  handleFile(file)
  isDragging.value = false
}

function onDragOver() {
  isDragging.value = true
}

function onDragLeave() {
  isDragging.value = false
}

function triggerFileInput() {
  fileInput.value.click()
}

function handleFileSelect(e) {
  const file = e.target.files[0]
  handleFile(file)
  e.target.value = null
}

async function uploadImage() {
  if (!selectedFile.value) return

  const formData = new FormData()
  formData.append('image', selectedFile.value)

  loading.value = true
  generatedMemeImage.value = null

  try {
    const response = await axios.post('http://localhost:9090/generate-meme', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    generatedMemeImage.value = response.data.meme_image_base64
  } catch (error) {
    console.error('Upload failed:', error)
    alert(`Failed to generate meme. Error: ${error.response?.data?.error || error.message}`)
    generatedMemeImage.value = null
  } finally {
    loading.value = false
  }
}

onUnmounted(() => {
  if (selectedImage.value) URL.revokeObjectURL(selectedImage.value)
})
</script>


<style scoped>
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.5s;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
}

.loader {
  border: 4px solid #cbd5e1; /* Light gray */
  border-top: 4px solid #3b82f6; /* Blue */
  border-radius: 50%;
  width: 32px;
  height: 32px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}


@keyframes gradientMove {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
}

.animated-gradient {
  background-size: 400% 400%;
  animation: gradientMove 15s ease infinite;
}
</style>
