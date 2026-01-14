<template>
  <div :class="['app', { 'dark': store.darkMode }]">
    <Header />
    <main class="main-content">
      <div class="container">
        <UploadSection v-if="!store.hasScore" />
        <template v-else>
          <ScoreInfo />
          <ControlPanel />
          <ScoreViewer v-if="store.hasFingering" />
        </template>
      </div>
    </main>
    <Footer />
    <ToastContainer />
  </div>
</template>

<script setup lang="ts">
import { useAppStore } from './stores/appStore';
import Header from './components/Header.vue';
import Footer from './components/Footer.vue';
import UploadSection from './components/UploadSection.vue';
import ScoreInfo from './components/ScoreInfo.vue';
import ControlPanel from './components/ControlPanel.vue';
import ScoreViewer from './components/ScoreViewer.vue';
import ToastContainer from './components/ToastContainer.vue';

const store = useAppStore();
</script>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--bg-primary);
  color: var(--text-primary);
  transition: background-color 0.3s, color 0.3s;
}

.main-content {
  flex: 1;
  padding: 2rem 1rem;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}
</style>
