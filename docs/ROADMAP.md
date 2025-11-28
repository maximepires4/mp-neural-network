# Project Roadmap

This roadmap outlines the planned improvements and features for `mp-neural-network`. It prioritizes performance optimization, stability, and architectural robustness.

---

## 🚀 High Priority (Performance & Core Architecture)

### **1. Robust Optimizer State (Fix `id()` issue)**
- [ ] Remove dependency on `id(param)` for tracking optimizer states (moments/velocities).
- [ ] Implement a stable parameter identification system (e.g., persistent UUIDs or structured naming).
- **Goal:** Fix issues where optimizer state is lost or mismatched after `load_model()`.

---

## 🛠 Medium Priority (Features & Usability)

### **2. User Experience Improvements**
- [ ] **Explicit Shape Checking:** Add clear error messages when connecting incompatible layers (e.g., "Dense(10) -> Dense(50): Mismatch").
- [ ] **Advanced Documentation:** Add Google-style docstrings to all public classes, exposing hidden features like "Smart Weights" and "He Initialization".

### **3. Functional Completeness**
- [ ] **Batch Prediction:** Update `model.predict()` to handle data in mini-batches internally to prevent OOM on large datasets.
- [ ] **Data Loaders:** Create a generator-based `DataLoader` for training on datasets larger than RAM.

---

## 🔮 Future / Exploration

- [ ] **RNN/LSTM Layers:** Support for sequential data.
- [ ] **GPU Acceleration:** Explore optional CuPy backend for NVIDIA GPU support.
- [ ] **Advanced Schedulers:** Implement Learning Rate Schedulers (ReduceLROnPlateau, CosineAnnealing).

---

## ✅ Completed Features (v1.0+)

### **Memory & Performance**
- [x] **Float32 Transition:** Global `DTYPE = np.float32` enforced to reduce RAM by ~50%.
- [x] **In-Place Operations:** Eliminated redundant copies in `train`, `predict`, and `evaluate`.
- [x] **Optimized Shuffle:** Shuffling indices instead of copying the full dataset.
- [x] **Lazy Metrics:** Metrics are computed only when necessary, speeding up training.

> **Note for Contributors:** Please refer to `docs/QUALITY_GUIDE.md` for contribution standards.
