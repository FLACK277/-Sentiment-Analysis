import { Canvas } from "@react-three/fiber";
import { Float, OrbitControls, RoundedBox, Text } from "@react-three/drei";
import type { AnalysisResponse, SentimentLabel } from "../types/sentiment";

const colorBySentiment: Record<SentimentLabel, string> = {
  positive: "#22c55e",
  negative: "#ef4444",
  neutral: "#60a5fa"
};

interface SentimentSceneProps {
  result: AnalysisResponse | null;
  loading: boolean;
}

const scoreLabel = (result: AnalysisResponse | null, loading: boolean): string => {
  if (loading) return "Analyzing...";
  if (!result) return "Awaiting input";
  return `${result.sentiment.toUpperCase()} (${(result.confidence * 100).toFixed(0)}%)`;
};

export const SentimentScene = ({ result, loading }: SentimentSceneProps) => {
  const color = result ? colorBySentiment[result.sentiment] : "#6366f1";

  return (
    <div className="scene-wrap">
      <Canvas camera={{ position: [0, 0.4, 4], fov: 45 }}>
        <ambientLight intensity={1} />
        <directionalLight position={[2, 3, 5]} intensity={2} />

        <Float speed={1.8} rotationIntensity={0.45} floatIntensity={0.95}>
          <RoundedBox args={[2.9, 1.8, 0.24]} radius={0.08} smoothness={5}>
            <meshStandardMaterial color={color} metalness={0.3} roughness={0.25} />
          </RoundedBox>

          <Text fontSize={0.16} position={[0, 0.25, 0.16]} maxWidth={2.6} textAlign="center">
            {scoreLabel(result, loading)}
          </Text>
          <Text fontSize={0.09} position={[0, -0.18, 0.16]} maxWidth={2.3} textAlign="center">
            {result?.explanation ?? "Type a review and run sentiment analysis"}
          </Text>
        </Float>

        <OrbitControls enablePan={false} minDistance={3} maxDistance={5} autoRotate autoRotateSpeed={0.7} />
      </Canvas>
    </div>
  );
};
