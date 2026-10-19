"use client";

import React, { useState, useEffect } from "react";
import Navbar from "../components/Navbar";
import ControlSidebar from "../components/ControlSidebar";
import ImageDetector from "../components/ImageDetector";
import VideoDetector from "../components/VideoDetector";
import CameraStream from "../components/CameraStream";

const DEFAULT_VOC_CLASSES = [
  "aeroplane", "bicycle", "bird", "boat", "bottle",
  "bus", "car", "cat", "chair", "cow",
  "diningtable", "dog", "horse", "motorbike", "person",
  "pottedplant", "sheep", "sofa", "train", "tvmonitor"
];

const DEFAULT_CLASS_COLORS = {
  aeroplane: "#38bdf8",
  bicycle: "#a855f7",
  bird: "#f43f5e",
  boat: "#06b6d4",
  bottle: "#14b8a6",
  bus: "#eab308",
  car: "#f97316",
  cat: "#ec4899",
  chair: "#8b5cf6",
  cow: "#84cc16",
  diningtable: "#64748b",
  dog: "#ef4444",
  horse: "#d97706",
  motorbike: "#6366f1",
  person: "#10b981",
  pottedplant: "#22c55e",
  sheep: "#0ea5e9",
  sofa: "#a1a1aa",
  train: "#f59e0b",
  tvmonitor: "#8b5cf6",
};

export default function Home() {
  const [activeTab, setActiveTab] = useState("image"); // 'image' | 'video' | 'camera'
  const [fps, setFps] = useState(10);
  const [confidence, setConfidence] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [selectedClasses, setSelectedClasses] = useState(DEFAULT_VOC_CLASSES);
  const [allClasses, setAllClasses] = useState(DEFAULT_VOC_CLASSES);
  const [classColors, setClassColors] = useState(DEFAULT_CLASS_COLORS);
  const [samples, setSamples] = useState([]);
  const [backendInfo, setBackendInfo] = useState(null);
  const [backendStatus, setBackendStatus] = useState("connecting");

  // Display toggles
  const [showLabels, setShowLabels] = useState(true);
  const [showConfidence, setShowConfidence] = useState(true);
  const [showCoordinates, setShowCoordinates] = useState(false);
  const [showGridGuide, setShowGridGuide] = useState(false);

  // Image Pyramid (Multi-Scale: Up + Down scale)
  const [usePyramid, setUsePyramid] = useState(true);
  const [pyramidType, setPyramidType] = useState("both"); // 'both' | 'up' | 'down' | 'none'

  const [streamStats, setStreamStats] = useState({ fps: 0 });

  // Load backend info & sample images
  useEffect(() => {
    async function initBackend() {
      try {
        setBackendStatus("connecting");
        const res = await fetch("/api/info");
        if (res.ok) {
          const data = await res.json();
          setBackendInfo(data);
          if (data.classes && data.classes.length > 0) {
            setAllClasses(data.classes);
            setSelectedClasses(data.classes);
          }
          if (data.class_colors) {
            setClassColors(data.class_colors);
          }
          setBackendStatus("connected");
        } else {
          setBackendStatus("disconnected");
        }

        const samplesRes = await fetch("/api/samples");
        if (samplesRes.ok) {
          const sData = await samplesRes.json();
          if (sData.samples) setSamples(sData.samples);
        }
      } catch (err) {
        console.warn("Backend connection check error:", err);
        setBackendStatus("disconnected");
      }
    }

    initBackend();
  }, []);

  return (
    <div style={styles.appContainer}>
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendInfo={backendInfo}
        backendStatus={backendStatus}
        streamStats={streamStats}
      />

      <main style={styles.mainContent}>
        <ControlSidebar
          activeTab={activeTab}
          fps={fps}
          setFps={setFps}
          confidence={confidence}
          setConfidence={setConfidence}
          iouThreshold={iouThreshold}
          setIouThreshold={setIouThreshold}
          selectedClasses={selectedClasses}
          setSelectedClasses={setSelectedClasses}
          allClasses={allClasses}
          classColors={classColors}
          showLabels={showLabels}
          setShowLabels={setShowLabels}
          showConfidence={showConfidence}
          setShowConfidence={setShowConfidence}
          showCoordinates={showCoordinates}
          setShowCoordinates={setShowCoordinates}
          showGridGuide={showGridGuide}
          setShowGridGuide={setShowGridGuide}
          usePyramid={usePyramid}
          setUsePyramid={setUsePyramid}
          pyramidType={pyramidType}
          setPyramidType={setPyramidType}
        />

        <section style={styles.tabWorkspace}>
          {activeTab === "image" && (
            <ImageDetector
              confidence={confidence}
              iouThreshold={iouThreshold}
              selectedClasses={selectedClasses}
              showLabels={showLabels}
              showConfidence={showConfidence}
              showCoordinates={showCoordinates}
              showGridGuide={showGridGuide}
              samples={samples}
              usePyramid={usePyramid}
              pyramidType={pyramidType}
            />
          )}

          {activeTab === "video" && (
            <VideoDetector
              fps={fps}
              confidence={confidence}
              iouThreshold={iouThreshold}
              selectedClasses={selectedClasses}
              showLabels={showLabels}
              showConfidence={showConfidence}
              showCoordinates={showCoordinates}
              onStatsUpdate={setStreamStats}
              usePyramid={usePyramid}
              pyramidType={pyramidType}
            />
          )}

          {activeTab === "camera" && (
            <CameraStream
              fps={fps}
              confidence={confidence}
              iouThreshold={iouThreshold}
              selectedClasses={selectedClasses}
              showLabels={showLabels}
              showConfidence={showConfidence}
              showCoordinates={showCoordinates}
              onStatsUpdate={setStreamStats}
              usePyramid={usePyramid}
              pyramidType={pyramidType}
            />
          )}
        </section>
      </main>

      <footer style={styles.footer}>
        <div style={styles.footerContent}>
          <div style={styles.footerCol}>
            <strong>OverFeat Detector Architecture:</strong> ResNet-18 Backbone (stride 32, window 224x224) • Conv Classification Head • Conv Box Regression Head • Mixed Precision AMP
          </div>
          <div style={styles.footerColRight}>
            PASCAL VOC 2007 (20 Object Classes) • FastAPI Backend • Next.js Frontend
          </div>
        </div>
      </footer>
    </div>
  );
}

const styles = {
  appContainer: {
    display: "flex",
    flexDirection: "column",
    minHeight: "100vh",
    backgroundColor: "#07090e",
    color: "#f8fafc",
  },
  mainContent: {
    display: "flex",
    flex: 1,
    overflow: "hidden",
  },
  tabWorkspace: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    padding: "16px 24px",
    overflowY: "auto",
    maxHeight: "calc(100vh - 72px)",
  },
  footer: {
    padding: "10px 24px",
    background: "rgba(10, 14, 23, 0.9)",
    borderTop: "1px solid rgba(255, 255, 255, 0.06)",
    fontSize: "11px",
    color: "#64748b",
  },
  footerContent: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  footerCol: {
    color: "#94a3b8",
  },
  footerColRight: {
    color: "#64748b",
  },
};
