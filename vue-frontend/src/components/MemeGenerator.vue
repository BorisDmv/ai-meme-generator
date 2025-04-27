<template>
  <div class="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
    <h1 class="text-3xl font-bold mb-6">AI Meme Generator</h1>

    <div
      class="border-4 border-dashed border-blue-400 rounded-2xl w-96 min-h-64 flex items-center justify-center cursor-pointer bg-white hover:bg-blue-50 transition mb-4"
      @dragover.prevent
      @drop.prevent="handleDrop"
      @click="triggerFileInput"
    >
      <p v-if="!selectedImage && !generatedMemeImage" class="text-gray-500 text-center px-4">
        Drag & drop an image here<br />or click to select
      </p>
      <img
        v-if="selectedImage && !generatedMemeImage && !loading"
        :src="selectedImage"
        alt="Uploaded Preview"
        class="max-h-60 object-contain rounded-lg"
      />
      <img
        v-if="generatedMemeImage && !loading"
        :src="generatedMemeImage"
        alt="Generated Meme"
        class="max-w-full max-h-96 object-contain rounded-lg shadow-lg"
      />
       <div v-if="loading" class="text-blue-500">Thinking... 🤔</div>
    </div>

     <input type="file" ref="fileInput" @change="handleFileSelect" accept="image/*" class="hidden" />

    <div v-if="selectedFile && !loading" class="mt-4">
       <button @click="uploadImage" class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition">
        {{ generatedMemeImage ? 'Generate Again' : 'Generate Meme' }}
      </button>
    </div>

    </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const selectedFile = ref(null)
const selectedImage = ref(null) // Stores the original image preview URL
const generatedMemeImage = ref(null) // Stores the base64 data URI of the final meme
const loading = ref(false)
const fileInput = ref(null) // Ref for the hidden file input

function handleFile(file) {
  if (file && file.type.startsWith('image/')) {
    selectedFile.value = file
    // Create object URL for the *original* preview
    if (selectedImage.value) {
        URL.revokeObjectURL(selectedImage.value) // Clean up previous object URL
    }
    selectedImage.value = URL.createObjectURL(file)
    generatedMemeImage.value = null // Clear previous generated meme
    // memeText.value = '' // No longer needed
  } else {
    alert('Please select an image file.')
    selectedFile.value = null
    selectedImage.value = null
    generatedMemeImage.value = null
  }
}

function handleDrop(e) {
  const file = e.dataTransfer.files[0]
  handleFile(file)
}

// --- Added for file input click ---
function triggerFileInput() {
    fileInput.value.click();
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    handleFile(file);
    // Reset the input value so the change event fires even if the same file is selected again
    e.target.value = null;
}
// --- End of added file input functions ---


async function uploadImage() {
  if (!selectedFile.value) return

  const formData = new FormData()
  formData.append('image', selectedFile.value)

  loading.value = true
  generatedMemeImage.value = null // Clear previous meme while loading new one
  // memeText.value = '' // No longer needed

  try {
    const response = await axios.post('http://localhost:9090/generate-meme', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    console.log('Response:', response.data)
    // Set the generatedMemeImage ref with the base64 data URI from the response
    generatedMemeImage.value = response.data.meme_image_base64
    // Optional: Log the received text too
    console.log("Generated caption:", response.data.caption);
    console.log("Generated meme text:", response.data.funny_meme_text);

  } catch (error) {
    console.error('Upload failed:', error)
    alert(`Failed to generate meme. Error: ${error.response?.data?.error || error.message || 'Unknown error'}`)
    generatedMemeImage.value = null // Ensure no broken image shows on error
  } finally {
    loading.value = false
  }
}

// Clean up object URL when component unmounts
import { onUnmounted } from 'vue';
onUnmounted(() => {
    if (selectedImage.value) {
        URL.revokeObjectURL(selectedImage.value);
    }
});

</script>

<style scoped>
/* Scoped styles remain the same */
/* No need for text shadow CSS as text is part of the image */
</style>