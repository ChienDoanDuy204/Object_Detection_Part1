"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";

export default function CameraStream({
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
  const [isStreaming, setIsStreaming] = useState(false);
  const [cameraDevices, setCameraDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [error, setError] = useState(null);
  const [detections, setDetections] = useState([]);
  const [timings, setTimings] = useState(null);
  const [realizedFps, setRealizedFps] = useState(0);
  const [roundTripMs, setRoundTripMs] = useState(0);
  const [wsConnected, setWsConnected] = useState(false);
  const [snapshotTaken, setSnapshotTaken] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const offscreenCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);
  const isProcessingRef = useRef(false);
  const frameCountRef = useRef(0);
  const lastFpsCalcRef = useRef(Date.now());
  const loopTimeoutRef = useRef(null);

  // Enumerate video devices
  useEffect(() => {
    async function getDevices() {
      try {
        if (!navigator?.mediaDevices?.enumerateDevices) return;
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices.filter((d) => d.kind === "videoinput");
        setCameraDevices(videoInputs);
        if (videoInputs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoInputs[0].deviceId);
        }
      } catch (err) {
        console.warn("Could not enumerate media devices:", err);
      }
    }
    getDevices();
  }, [selectedDeviceId]);

  // Connect WebSocket
  useEffect(() => {
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsHost = window.location.hostname === "localhost" ? "localhost:8000" : window.location.host;
    const wsUrl = `${wsProtocol}//${wsHost}/ws/stream`;

    let socket;
    try {
      socket = new WebSocket(wsUrl);
      wsRef.current = socket;

      socket.onopen = () => {
        setWsConnected(true);
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.status === "success") {
            setDetections(data.detections || []);
            setTimings(data.timings_ms || null);

            if (data.client_timestamp) {
              const rtt = Math.round(performance.now() - data.client_timestamp);
              setRoundTripMs(rtt);
            }

            frameCountRef.current += 1;
            const now = Date.now();
            const elapsed = now - lastFpsCalcRef.current;
            if (elapsed >= 1000) {
              const currentFps = Math.round((frameCountRef.current * 1000) / elapsed);
              setRealizedFps(currentFps);
              frameCountRef.current = 0;
              lastFpsCalcRef.current = now;
              onStatsUpdate?.({ fps: currentFps });
            }
          }
        } catch (err) {
          console.error("WS message parse error:", err);
        } finally {
          isProcessingRef.current = false;
        }
      };

      socket.onerror = () => {
        setWsConnected(false);
      };

      socket.onclose = () => {
        setWsConnected(false);
      };
    } catch (err) {
      console.warn("WebSocket connection failed, will use HTTP fallback:", err);
    }

    return () => {
      if (socket) socket.close();
    };
  }, [onStatsUpdate]);

  // Start Camera Stream
  const startCamera = async () => {
    setError(null);
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }

      const constraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
          : { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsStreaming(true);
      }
    } catch (err) {
      setError("Unable to access camera: " + err.message + ". Please ensure camera permissions are granted.");
      setIsStreaming(false);
    }
  };

  // Stop Camera Stream
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setDetections([]);
    setRealizedFps(0);
    onStatsUpdate?.({ fps: 0 });
    if (loopTimeoutRef.current) clearTimeout(loopTimeoutRef.current);
  };

  // Frame Capture Loop regulated by user-selected FPS!
  const captureAndSendFrame = useCallback(async () => {
    if (!isStreaming || !videoRef.current) return;
    const video = videoRef.current;
    if (video.readyState < 2) return;

    // Throttle: ensure previous frame finished and interval elapsed
    if (!isProcessingRef.current) {
      isProcessingRef.current = true;
      const clientTimestamp = performance.now();

      if (!offscreenCanvasRef.current) {
        offscreenCanvasRef.current = document.createElement("canvas");
      }
      const offscreen = offscreenCanvasRef.current;
      const targetW = video.videoWidth || 640;
      const targetH = video.videoHeight || 480;

      if (offscreen.width !== targetW) offscreen.width = targetW;
      if (offscreen.height !== targetH) offscreen.height = targetH;

      const offCtx = offscreen.getContext("2d");
      offCtx.drawImage(video, 0, 0, targetW, targetH);

      const base64Data = offscreen.toDataURL("image/jpeg", 0.65);

      // Check if WebSocket is ready
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            frame: base64Data,
            confidence_threshold: confidence,
            iou_threshold: iouThreshold,
            allowed_classes: selectedClasses.length > 0 ? selectedClasses : null,
            max_dim: 448,
            use_pyramid: usePyramid,
            pyramid_type: pyramidType,
            client_timestamp: clientTimestamp,
          })
        );
      } else {
        // Fallback: HTTP POST
        try {
          const res = await fetch("/api/predict/frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              frame: base64Data,
              confidence_threshold: confidence,
              iou_threshold: iouThreshold,
              allowed_classes: selectedClasses.length > 0 ? selectedClasses : null,
              max_dim: 448,
              use_pyramid: usePyramid,
              pyramid_type: pyramidType,
            }),
          });

          if (res.ok) {
            const data = await res.json();
            setDetections(data.detections || []);
            setTimings(data.timings_ms || null);
            setRoundTripMs(Math.round(performance.now() - clientTimestamp));

            frameCountRef.current += 1;
            const now = Date.now();
            const elapsed = now - lastFpsCalcRef.current;
            if (elapsed >= 1000) {
              const currentFps = Math.round((frameCountRef.current * 1000) / elapsed);
              setRealizedFps(currentFps);
              frameCountRef.current = 0;
              lastFpsCalcRef.current = now;
              onStatsUpdate?.({ fps: currentFps });
            }
          }
        } catch (err) {
          console.error("HTTP fallback frame error:", err);
        } finally {
          isProcessingRef.current = false;
        }
      }
    }
  }, [isStreaming, confidence, iouThreshold, selectedClasses, onStatsUpdate]);

  // Regulated timer loop based on FPS
  useEffect(() => {
    if (!isStreaming) return;

    const intervalMs = Math.max(33, Math.round(1000 / fps)); // FPS controller!
    const tick = () => {
      captureAndSendFrame();
      loopTimeoutRef.current = setTimeout(tick, intervalMs);
    };

    loopTimeoutRef.current = setTimeout(tick, intervalMs);
    return () => {
      if (loopTimeoutRef.current) clearTimeout(loopTimeoutRef.current);
    };
  }, [isStreaming, fps, captureAndSendFrame]);

  // Draw overlay canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    if (video.videoWidth && canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach((det) => {
      const [xmin, ymin, xmax, ymax] = det.box;
      const w = xmax - xmin;
      const h = ymax - ymin;
      const color = det.color || "#38bdf8";

      // Semi-transparent box
      ctx.fillStyle = `${color}25`;
      ctx.fillRect(xmin, ymin, w, h);

      // Border
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
  }, [detections, showLabels, showConfidence]);

  const captureSnapshot = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const snapCanvas = document.createElement("canvas");
    snapCanvas.width = canvas.width;
    snapCanvas.height = canvas.height;
    const snapCtx = snapCanvas.getContext("2d");

    // Draw video then overlay canvas
    snapCtx.drawImage(video, 0, 0);
    snapCtx.drawImage(canvas, 0, 0);

    const link = document.createElement("a");
    link.download = `camera_snapshot_${Date.now()}.png`;
    link.href = snapCanvas.toDataURL("image/png");
    link.click();

    setSnapshotTaken(true);
    setTimeout(() => setSnapshotTaken(false), 2000);
  };

  return (
    <div style={styles.container}>
      {/* Top Controls Bar */}
      <div style={styles.controlBar}>
        <div style={styles.controlGroup}>
          {!isStreaming ? (
            <button style={styles.startBtn} onClick={startCamera}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <polygon points="5 3 19 12 5 21 5 3"/>
              </svg>
              <span>Start Camera Stream</span>
            </button>
          ) : (
            <button style={styles.stopBtn} onClick={stopCamera}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <rect x="6" y="6" width="12" height="12"/>
              </svg>
              <span>Stop Camera</span>
            </button>
          )}

          {cameraDevices.length > 1 && (
            <select
              style={styles.deviceSelect}
              value={selectedDeviceId}
              onChange={(e) => {
                setSelectedDeviceId(e.target.value);
                if (isStreaming) {
                  stopCamera();
                  setTimeout(startCamera, 300);
                }
              }}
            >
              {cameraDevices.map((dev, i) => (
                <option key={dev.deviceId || i} value={dev.deviceId}>
                  {dev.label || `Camera ${i + 1}`}
                </option>
              ))}
            </select>
          )}
        </div>

        {isStreaming && (
          <div style={styles.streamingMeta}>
            <span style={styles.fpsControlInfo}>
              Throttle Rate: <strong style={{ color: "#38bdf8" }}>{fps} FPS</strong> (Interval ~{Math.round(1000/fps)}ms)
            </span>
            <button style={styles.snapshotBtn} onClick={captureSnapshot}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                <circle cx="12" cy="13" r="4"/>
              </svg>
              <span>{snapshotTaken ? "Saved!" : "Capture Snapshot"}</span>
            </button>
          </div>
        )}
      </div>

      {error && (
        <div style={styles.errorBanner}>
          <span>{error}</span>
        </div>
      )}

      {/* Main Workspace */}
      <div style={styles.workspace}>
        {/* Camera Stage */}
        <div style={styles.cameraWrapper}>
          {!isStreaming ? (
            <div style={styles.idleState} onClick={startCamera}>
              <div style={styles.cameraIconCircle}>
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="1.5">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                  <circle cx="12" cy="13" r="4"/>
                </svg>
              </div>
              <span style={styles.idleTitle}>Camera is Currently Inactive</span>
              <span style={styles.idleSubtitle}>
                Click here or &apos;Start Camera Stream&apos; above to begin real-time detection
              </span>
            </div>
          ) : (
            <div style={styles.activeStage}>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={styles.videoFeed}
              />
              <canvas ref={canvasRef} style={styles.canvasOverlay} />

              {/* Stream Telemetry HUD */}
              <div style={styles.telemetryHUD}>
                <span style={styles.livePulseDot} />
                <span>Stream: {realizedFps} / {fps} FPS</span>
                <span>•</span>
                <span>Infer: {timings?.inference ?? 0}ms</span>
                <span>•</span>
                <span>RTT: {roundTripMs}ms</span>
                <span>•</span>
                <span style={{ color: wsConnected ? "#10b981" : "#f59e0b" }}>
                  {wsConnected ? "WebSocket" : "HTTP"}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Live Detections Panel */}
        <div style={styles.detectionsSide}>
          <div style={styles.metricsCard}>
            <span style={styles.cardHeaderTitle}>Live Performance Telemetry</span>
            <div style={styles.telemetryGrid}>
              <div style={styles.teleItem}>
                <span style={styles.teleLabel}>Processed FPS</span>
                <span style={styles.teleValCyan}>{realizedFps} fps</span>
              </div>
              <div style={styles.teleItem}>
                <span style={styles.teleLabel}>Target Limit</span>
                <span style={styles.teleVal}>{fps} fps</span>
              </div>
              <div style={styles.teleItem}>
                <span style={styles.teleLabel}>Model Latency</span>
                <span style={styles.teleValEmerald}>{timings?.inference ?? 0} ms</span>
              </div>
              <div style={styles.teleItem}>
                <span style={styles.teleLabel}>Active Objects</span>
                <span style={styles.teleValViolet}>{detections.length}</span>
              </div>
            </div>
          </div>

          <div style={styles.detectionsListCard}>
            <div style={styles.listHeader}>
              <span style={styles.cardHeaderTitle}>Real-Time Detections</span>
              <span style={styles.criteriaTag}>
                &gt;{Math.round(confidence * 100)}% conf
              </span>
            </div>

            {detections.length === 0 ? (
              <div style={styles.emptyDets}>
                {isStreaming
                  ? "Point camera at objects to detect (Person, Bird, Car, Dog, etc.)"
                  : "Start stream to detect objects"}
              </div>
            ) : (
              <div style={styles.detsScroll}>
                {detections.map((det, i) => (
                  <div key={i} style={styles.detItem}>
                    <div style={styles.detItemLeft}>
                      <span
                        style={{
                          ...styles.detPillDot,
                          backgroundColor: det.color || "#38bdf8",
                        }}
                      />
                      <strong style={styles.detTitle}>{det.class_name}</strong>
                    </div>
                    <span style={{ ...styles.detScore, color: det.color || "#38bdf8" }}>
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
  controlBar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    background: "rgba(18, 24, 38, 0.6)",
    padding: "10px 18px",
    borderRadius: "12px",
    border: "1px solid rgba(255, 255, 255, 0.07)",
  },
  controlGroup: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
  },
  startBtn: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "8px 18px",
    background: "linear-gradient(135deg, #059669, #10b981)",
    color: "#ffffff",
    border: "none",
    borderRadius: "8px",
    fontSize: "13px",
    fontWeight: "600",
    cursor: "pointer",
    boxShadow: "0 2px 10px rgba(16, 185, 129, 0.25)",
  },
  stopBtn: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "8px 18px",
    background: "linear-gradient(135deg, #dc2626, #ef4444)",
    color: "#ffffff",
    border: "none",
    borderRadius: "8px",
    fontSize: "13px",
    fontWeight: "600",
    cursor: "pointer",
  },
  deviceSelect: {
    padding: "7px 12px",
    background: "rgba(8, 11, 17, 0.8)",
    border: "1px solid rgba(255, 255, 255, 0.1)",
    borderRadius: "8px",
    color: "#f8fafc",
    fontSize: "12.5px",
    outline: "none",
    cursor: "pointer",
  },
  streamingMeta: {
    display: "flex",
    alignItems: "center",
    gap: "14px",
  },
  fpsControlInfo: {
    fontSize: "12px",
    color: "#94a3b8",
  },
  snapshotBtn: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "7px 14px",
    background: "rgba(255, 255, 255, 0.06)",
    border: "1px solid rgba(255, 255, 255, 0.12)",
    borderRadius: "8px",
    color: "#cbd5e1",
    fontSize: "12px",
    fontWeight: "500",
    cursor: "pointer",
  },
  errorBanner: {
    background: "rgba(239, 68, 68, 0.2)",
    border: "1px solid rgba(239, 68, 68, 0.4)",
    color: "#fca5a5",
    padding: "10px 16px",
    borderRadius: "8px",
    fontSize: "12.5px",
  },
  workspace: {
    display: "flex",
    gap: "16px",
    flex: 1,
    minHeight: "520px",
  },
  cameraWrapper: {
    flex: 2,
    display: "flex",
    flexDirection: "column",
    background: "rgba(14, 19, 31, 0.6)",
    borderRadius: "16px",
    border: "1px solid rgba(255, 255, 255, 0.07)",
    padding: "14px",
  },
  idleState: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: "12px",
    background: "#080b11",
    borderRadius: "12px",
    border: "2px dashed rgba(255, 255, 255, 0.1)",
    cursor: "pointer",
    minHeight: "440px",
  },
  cameraIconCircle: {
    width: "72px",
    height: "72px",
    borderRadius: "50%",
    background: "rgba(6, 182, 212, 0.1)",
    border: "1px solid rgba(6, 182, 212, 0.25)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  idleTitle: {
    fontSize: "15px",
    fontWeight: "600",
    color: "#f8fafc",
  },
  idleSubtitle: {
    fontSize: "12.5px",
    color: "#64748b",
  },
  activeStage: {
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
  videoFeed: {
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
  telemetryHUD: {
    position: "absolute",
    top: "14px",
    left: "14px",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    background: "rgba(8, 11, 17, 0.8)",
    backdropFilter: "blur(8px)",
    border: "1px solid rgba(255, 255, 255, 0.1)",
    padding: "5px 14px",
    borderRadius: "20px",
    fontSize: "11.5px",
    fontWeight: "600",
    color: "#e2e8f0",
  },
  livePulseDot: {
    width: "7px",
    height: "7px",
    borderRadius: "50%",
    backgroundColor: "#ef4444",
    boxShadow: "0 0 6px #ef4444",
  },
  detectionsSide: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  metricsCard: {
    background: "rgba(18, 24, 38, 0.7)",
    borderRadius: "12px",
    border: "1px solid rgba(255, 255, 255, 0.07)",
    padding: "14px",
  },
  cardHeaderTitle: {
    fontSize: "12.5px",
    fontWeight: "700",
    color: "#cbd5e1",
    display: "block",
    marginBottom: "10px",
  },
  telemetryGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, 1fr)",
    gap: "8px",
  },
  teleItem: {
    background: "rgba(8, 11, 17, 0.5)",
    padding: "8px 10px",
    borderRadius: "8px",
    display: "flex",
    flexDirection: "column",
    gap: "2px",
  },
  teleLabel: {
    fontSize: "10.5px",
    color: "#94a3b8",
  },
  teleValCyan: {
    fontSize: "16px",
    fontWeight: "800",
    color: "#38bdf8",
  },
  teleValEmerald: {
    fontSize: "16px",
    fontWeight: "800",
    color: "#10b981",
  },
  teleValViolet: {
    fontSize: "16px",
    fontWeight: "800",
    color: "#a855f7",
  },
  teleVal: {
    fontSize: "15px",
    fontWeight: "700",
    color: "#f8fafc",
  },
  detectionsListCard: {
    flex: 1,
    background: "rgba(14, 19, 31, 0.6)",
    borderRadius: "14px",
    border: "1px solid rgba(255, 255, 255, 0.07)",
    padding: "14px",
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },
  listHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  criteriaTag: {
    fontSize: "11px",
    color: "#64748b",
  },
  emptyDets: {
    fontSize: "12px",
    color: "#64748b",
    textAlign: "center",
    marginTop: "24px",
    lineHeight: "1.5",
  },
  detsScroll: {
    display: "flex",
    flexDirection: "column",
    gap: "6px",
    overflowY: "auto",
    maxHeight: "300px",
  },
  detItem: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "8px 12px",
    background: "rgba(18, 24, 38, 0.6)",
    borderRadius: "8px",
    border: "1px solid rgba(255, 255, 255, 0.05)",
  },
  detItemLeft: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  detPillDot: {
    width: "7px",
    height: "7px",
    borderRadius: "50%",
  },
  detTitle: {
    fontSize: "12.5px",
    textTransform: "capitalize",
    color: "#f8fafc",
  },
  detScore: {
    fontSize: "12px",
    fontWeight: "700",
  },
};
