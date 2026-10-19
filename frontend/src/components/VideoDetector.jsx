"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";

export default function VideoDetector({
  fps,
  confidence,
  iouThreshold,
  selectedClasses,
  showLabels,
  showConfidence,
  showCoordinates,
  onStatsUpdate,
  usePyramid = false,
  pyramidType = "both",
}) {
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [detections, setDetections] = useState([]);
  const [timings, setTimings] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [batchProgress, setBatchProgress] = useState(null);
  const [annotatedVideoUrl, setAnnotatedVideoUrl] = useState(null);
  const [realizedFps, setRealizedFps] = useState(0);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const offscreenCanvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const lastProcessedTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const lastFpsCalcRef = useRef(Date.now());
  const fileInputRef = useRef(null);

  const handleVideoUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
      const url = URL.createObjectURL(file);
      setVideoFile(file);
      setVideoUrl(url);
      setDetections([]);
      setAnnotatedVideoUrl(null);
      setCurrentTime(0);
      setIsPlaying(false);
    }
  };

  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;
    setDuration(video.duration || 0);

    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 360;
    }

    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement("canvas");
    }
    offscreenCanvasRef.current.width = video.videoWidth || 640;
    offscreenCanvasRef.current.height = video.videoHeight || 360;
  };

  // Process single frame from video
  const processFrame = useCallback(async () => {
    const video = videoRef.current;
    if (!video || video.paused || video.ended || isProcessing) return;

    const now = performance.now();
    const frameIntervalMs = 1000 / fps; // Controlled by the FPS slider!

    if (now - lastProcessedTimeRef.current >= frameIntervalMs) {
      lastProcessedTimeRef.current = now;
      setIsProcessing(true);

      try {
        const offscreen = offscreenCanvasRef.current;
        if (!offscreen) return;
        const offCtx = offscreen.getContext("2d");
        offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);

        // Convert to base64 jpeg
        const base64Data = offscreen.toDataURL("image/jpeg", 0.7);

        const res = await fetch("/api/predict/frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            frame: base64Data,
            confidence_threshold: confidence,
            iou_threshold: iouThreshold,
            allowed_classes: selectedClasses.length > 0 ? selectedClasses : null,
            max_dim: 480,
            use_pyramid: usePyramid,
            pyramid_type: pyramidType,
          }),
        });

        if (res.ok) {
          const data = await res.json();
          setDetections(data.detections || []);
          setTimings(data.timings_ms || null);

          // Calculate realized FPS
          frameCountRef.current += 1;
          const timeSinceLastFps = Date.now() - lastFpsCalcRef.current;
          if (timeSinceLastFps >= 1000) {
            const currentRealizedFps = Math.round((frameCountRef.current * 1000) / timeSinceLastFps);
            setRealizedFps(currentRealizedFps);
            frameCountRef.current = 0;
            lastFpsCalcRef.current = Date.now();
            onStatsUpdate?.({ fps: currentRealizedFps });
          }
        }
      } catch (err) {
        console.error("Frame processing error:", err);
      } finally {
        setIsProcessing(false);
      }
    }
  }, [fps, confidence, iouThreshold, selectedClasses, isProcessing, onStatsUpdate]);

  // Main playback loop
  useEffect(() => {
    if (!isPlaying) {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      return;
    }

    const loop = () => {
      const video = videoRef.current;
      if (video) {
        setCurrentTime(video.currentTime);
        processFrame();
      }
      animationFrameRef.current = requestAnimationFrame(loop);
    };

    animationFrameRef.current = requestAnimationFrame(loop);
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [isPlaying, processFrame]);

  // Draw bounding boxes on canvas overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach((det) => {
      const [xmin, ymin, xmax, ymax] = det.box;
      const w = xmax - xmin;
      const h = ymax - ymin;
      const color = det.color || "#38bdf8";

      // Draw box
      ctx.fillStyle = `${color}25`;
      ctx.fillRect(xmin, ymin, w, h);

      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(xmin, ymin, w, h);

      // Label & Score
      if (showLabels || showConfidence) {
        let labelText = "";
        if (showLabels) labelText += det.class_name.toUpperCase();
        if (showConfidence) {
          labelText += (showLabels ? " " : "") + `${Math.round(det.confidence * 100)}%`;
        }

        ctx.font = "bold 13px sans-serif";
        const textMetrics = ctx.measureText(labelText);
        const tagHeight = 22;
        const tagWidth = textMetrics.width + 12;
        const tagY = Math.max(0, ymin - tagHeight - 2);

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(xmin, tagY, tagWidth, tagHeight, [4, 4, 0, 0]);
        ctx.fill();

        ctx.fillStyle = "#ffffff";
        ctx.textBaseline = "middle";
        ctx.fillText(labelText, xmin + 6, tagY + tagHeight / 2);
      }
    });
  }, [detections, showLabels, showConfidence, showCoordinates]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  };

  const handleSeek = (e) => {
    const target = Number(e.target.value);
    const video = videoRef.current;
    if (video) {
      video.currentTime = target;
      setCurrentTime(target);
    }
  };

  // Batch process video on backend
  const handleBatchProcess = async () => {
    if (!videoFile) return;
    setBatchProgress("Uploading & processing on GPU at " + fps + " FPS...");
    try {
      const formData = new FormData();
      formData.append("file", videoFile);
      formData.append("target_fps", fps);
      formData.append("confidence_threshold", confidence);
      formData.append("iou_threshold", iouThreshold);
      formData.append("annotate", "true");
      formData.append("use_pyramid", usePyramid ? "true" : "false");
      formData.append("pyramid_type", pyramidType);
      if (selectedClasses.length > 0) {
        formData.append("allowed_classes", JSON.stringify(selectedClasses));
      }

      const res = await fetch("/api/predict/video", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Failed to process video on server");
      }

      const data = await res.json();
      setBatchProgress(`Processed ${data.video_info.processed_frames} frames successfully!`);
      if (data.annotated_video_url) {
        setAnnotatedVideoUrl(data.annotated_video_url);
      }
    } catch (err) {
      setBatchProgress("Error: " + err.message);
    }
  };

  return (
    <div style={styles.container}>
      {/* Upload Banner */}
      <div style={styles.topControlBar}>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          style={{ display: "none" }}
          onChange={handleVideoUpload}
        />
        <button
          style={styles.uploadBtn}
          onClick={() => fileInputRef.current?.click()}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <span>{videoFile ? "Change Video File" : "Upload Video File"}</span>
        </button>

        {videoFile && (
          <div style={styles.fileDetails}>
            <span style={styles.fileName}>{videoFile.name}</span>
            <span style={styles.fpsNotice}>
              Target Sampling: <strong style={{ color: "#38bdf8" }}>{fps} FPS</strong>
            </span>
          </div>
        )}

        {videoFile && (
          <button style={styles.batchBtn} onClick={handleBatchProcess}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="5 3 19 12 5 21 5 3"/>
            </svg>
            <span>Batch Process Full Video</span>
          </button>
        )}
      </div>

      {batchProgress && (
        <div style={styles.progressBanner}>
          <span>{batchProgress}</span>
          {annotatedVideoUrl && (
            <a
              href={annotatedVideoUrl}
              download
              style={styles.downloadLink}
            >
              Download Annotated Video
            </a>
          )}
        </div>
      )}

      {/* Main Video Display & Overlay */}
      <div style={styles.workspace}>
        <div style={styles.playerWrapper}>
          {!videoUrl ? (
            <div
              style={styles.emptyPlayer}
              onClick={() => fileInputRef.current?.click()}
            >
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="1.5">
                <polygon points="23 7 16 12 23 17 23 7"/>
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
              </svg>
              <span style={styles.emptyText}>Upload an MP4 / WebM video to start real-time detection</span>
              <span style={styles.emptySubtext}>Model processes frames dynamically at your chosen {fps} FPS rate</span>
            </div>
          ) : (
            <div style={styles.videoStage}>
              <video
                ref={videoRef}
                src={videoUrl}
                style={styles.videoElement}
                playsInline
                onLoadedMetadata={handleLoadedMetadata}
                onEnded={() => setIsPlaying(false)}
              />
              <canvas ref={canvasRef} style={styles.canvasOverlay} />

              {/* Real-time HUD pill */}
              <div style={styles.hudOverlay}>
                <span style={styles.hudLiveDot} />
                <span>Rate: {realizedFps} / {fps} FPS</span>
                <span>•</span>
                <span>Infer: {timings?.inference ?? 0}ms</span>
                <span>•</span>
                <span>Objects: {detections.length}</span>
              </div>
            </div>
          )}

          {/* Timeline and Playback Controls */}
          {videoUrl && (
            <div style={styles.controlsBar}>
              <button style={styles.playPauseBtn} onClick={togglePlay}>
                {isPlaying ? (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <rect x="6" y="4" width="4" height="16"/>
                    <rect x="14" y="4" width="4" height="16"/>
                  </svg>
                ) : (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                  </svg>
                )}
              </button>

              <span style={styles.timeDisplay}>
                {Math.floor(currentTime)}s / {Math.floor(duration)}s
              </span>

              <input
                type="range"
                min="0"
                max={duration || 1}
                step="0.1"
                value={currentTime}
                onChange={handleSeek}
                style={styles.timelineSlider}
              />
            </div>
          )}
        </div>

        {/* Live Detections Sidebar for Video */}
        <div style={styles.sideStats}>
          <div style={styles.statsCard}>
            <span style={styles.statsTitle}>Video Stream Metrics</span>
            <div style={styles.statsRow}>
              <span style={styles.statLabel}>Processing FPS</span>
              <span style={styles.statValCyan}>{realizedFps} fps</span>
            </div>
            <div style={styles.statsRow}>
              <span style={styles.statLabel}>Target Throttle</span>
              <span style={styles.statVal}>{fps} fps ({Math.round(1000/fps)}ms)</span>
            </div>
            <div style={styles.statsRow}>
              <span style={styles.statLabel}>Inference Latency</span>
              <span style={styles.statValEmerald}>{timings?.inference ?? 0} ms</span>
            </div>
            <div style={styles.statsRow}>
              <span style={styles.statLabel}>Total Latency</span>
              <span style={styles.statVal}>{timings?.total ?? 0} ms</span>
            </div>
          </div>

          <div style={styles.detectionsBox}>
            <span style={styles.statsTitle}>Current Frame Objects ({detections.length})</span>
            {detections.length === 0 ? (
              <span style={styles.emptyDets}>
                {isPlaying ? "Searching for objects satisfying criteria..." : "Play video to process frames"}
              </span>
            ) : (
              <div style={styles.detsList}>
                {detections.map((det, i) => (
                  <div key={i} style={styles.detRow}>
                    <div style={styles.detLeft}>
                      <span
                        style={{
                          ...styles.detDot,
                          backgroundColor: det.color || "#38bdf8",
                        }}
                      />
                      <strong style={styles.detName}>{det.class_name}</strong>
                    </div>
                    <span style={{ ...styles.detPct, color: det.color || "#38bdf8" }}>
                      {Math.round(det.confidence * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "14px",
    flex: 1,
  },
  topControlBar: {
    display: "flex",
    alignItems: "center",
    gap: "14px",
    background: "rgba(18, 24, 38, 0.6)",
    padding: "10px 18px",
    borderRadius: "12px",
    border: "1px solid rgba(255, 255, 255, 0.07)",
  },
  uploadBtn: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "8px 16px",
    background: "linear-gradient(135deg, #0284c7, #0369a1)",
    color: "#ffffff",
    border: "none",
    borderRadius: "8px",
    fontSize: "13px",
    fontWeight: "600",
    cursor: "pointer",
  },
  fileDetails: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    flex: 1,
  },
  fileName: {
    fontSize: "13px",
    fontWeight: "500",
    color: "#f8fafc",
  },
  fpsNotice: {
    fontSize: "12px",
    color: "#94a3b8",
  },
  batchBtn: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "8px 14px",
    background: "rgba(139, 92, 246, 0.2)",
    color: "#c084fc",
    border: "1px solid rgba(139, 92, 246, 0.4)",
    borderRadius: "8px",
    fontSize: "12.5px",
    fontWeight: "600",
    cursor: "pointer",
  },
  progressBanner: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    background: "rgba(16, 185, 129, 0.15)",
    border: "1px solid rgba(16, 185, 129, 0.3)",
    color: "#6ee7b7",
    padding: "10px 18px",
    borderRadius: "8px",
    fontSize: "12.5px",
  },
  downloadLink: {
    color: "#ffffff",
    background: "#059669",
    padding: "4px 10px",
    borderRadius: "6px",
    textDecoration: "none",
    fontWeight: "600",
  },
  workspace: {
    display: "flex",
    gap: "16px",
    flex: 1,
    minHeight: "520px",
  },
  playerWrapper: {
    flex: 2,
    display: "flex",
    flexDirection: "column",
    gap: "10px",
    background: "rgba(14, 19, 31, 0.6)",
    borderRadius: "16px",
    border: "1px solid rgba(255, 255, 255, 0.07)",
    padding: "14px",
  },
  emptyPlayer: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: "10px",
    background: "#080b11",
    borderRadius: "12px",
    border: "2px dashed rgba(255, 255, 255, 0.1)",
    cursor: "pointer",
    minHeight: "440px",
  },
  emptyText: {
    fontSize: "14px",
    fontWeight: "600",
    color: "#cbd5e1",
  },
  emptySubtext: {
    fontSize: "12px",
    color: "#64748b",
  },
  videoStage: {
    position: "relative",
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#000000",
    borderRadius: "12px",
    overflow: "hidden",
    minHeight: "440px",
  },
  videoElement: {
    maxWidth: "100%",
    maxHeight: "100%",
    borderRadius: "8px",
  },
  canvasOverlay: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    pointerEvents: "none",
  },
  hudOverlay: {
    position: "absolute",
    top: "14px",
    left: "14px",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    background: "rgba(8, 11, 17, 0.75)",
    backdropFilter: "blur(8px)",
    border: "1px solid rgba(255, 255, 255, 0.1)",
    padding: "4px 12px",
    borderRadius: "20px",
    fontSize: "11.5px",
    fontWeight: "600",
    color: "#e2e8f0",
  },
  hudLiveDot: {
    width: "6px",
    height: "6px",
    borderRadius: "50%",
    backgroundColor: "#10b981",
    boxShadow: "0 0 6px #10b981",
  },
  controlsBar: {
    display: "flex",
    alignItems: "center",
    gap: "14px",
    padding: "8px 14px",
    background: "rgba(18, 24, 38, 0.8)",
    borderRadius: "10px",
    border: "1px solid rgba(255, 255, 255, 0.08)",
  },
  playPauseBtn: {
    width: "36px",
    height: "36px",
    borderRadius: "50%",
    background: "linear-gradient(135deg, #0284c7, #0369a1)",
    border: "none",
    color: "#ffffff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
  },
  timeDisplay: {
    fontSize: "12px",
    fontFamily: "monospace",
    color: "#94a3b8",
    whiteSpace: "nowrap",
  },
  timelineSlider: {
    flex: 1,
    cursor: "pointer",
  },
  sideStats: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  statsCard: {
    background: "rgba(18, 24, 38, 0.7)",
    borderRadius: "12px",
    border: "1px solid rgba(255, 255, 255, 0.07)",
    padding: "14px",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  statsTitle: {
    fontSize: "12.5px",
    fontWeight: "700",
    color: "#cbd5e1",
    marginBottom: "4px",
  },
  statsRow: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: "12px",
  },
  statLabel: {
    color: "#94a3b8",
  },
  statValCyan: {
    fontWeight: "700",
    color: "#38bdf8",
  },
  statValEmerald: {
    fontWeight: "700",
    color: "#10b981",
  },
  statVal: {
    fontWeight: "600",
    color: "#e2e8f0",
  },
  detectionsBox: {
    flex: 1,
    background: "rgba(14, 19, 31, 0.6)",
    borderRadius: "14px",
    border: "1px solid rgba(255, 255, 255, 0.07)",
    padding: "14px",
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },
  emptyDets: {
    fontSize: "12px",
    color: "#64748b",
    textAlign: "center",
    marginTop: "20px",
  },
  detsList: {
    display: "flex",
    flexDirection: "column",
    gap: "6px",
    overflowY: "auto",
    maxHeight: "300px",
  },
  detRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "8px 10px",
    background: "rgba(18, 24, 38, 0.6)",
    borderRadius: "8px",
    border: "1px solid rgba(255, 255, 255, 0.05)",
  },
  detLeft: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  detDot: {
    width: "7px",
    height: "7px",
    borderRadius: "50%",
  },
  detName: {
    fontSize: "12.5px",
    textTransform: "capitalize",
    color: "#f8fafc",
  },
  detPct: {
    fontSize: "12px",
    fontWeight: "700",
  },
};
