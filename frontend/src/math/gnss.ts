/**
 * FlightML GNSS Kinematics & EW Shield (TypeScript).
 */

export function teleportationGroundspeedKts(distanceNm: number, deltaTimeS: number): number {
  if (deltaTimeS <= 0) return 0.0;
  return distanceNm / (deltaTimeS / 3600.0);
}

export function verticalRateFpm(deltaAltitudeFt: number, deltaTimeS: number): number {
  if (deltaTimeS <= 0) return 0.0;
  return Math.abs(deltaAltitudeFt) / (deltaTimeS / 60.0);
}

export function isTerminalVectoringZone(
  distToAdepNm: number,
  distToAdesNm: number,
  terminalRadiusNm: number = 50.0
): boolean {
  return distToAdepNm <= terminalRadiusNm || distToAdesNm <= terminalRadiusNm;
}

export function gnssCompositeSpoofingScore(
  teleportCount: number,
  verticalSpikeCount: number,
  altitudeGlitchCount: number,
  trackJitterCount: number,
  inHotspot: boolean = false
): number {
  let score = 0.0;
  score += Math.min(70.0, teleportCount * 35.0);
  score += Math.min(60.0, verticalSpikeCount * 30.0);
  score += Math.min(50.0, altitudeGlitchCount * 25.0);
  score += Math.min(20.0, trackJitterCount * 10.0);

  if (inHotspot && (teleportCount > 0 || verticalSpikeCount > 0 || altitudeGlitchCount > 0)) {
    score += 15.0;
  }
  return Math.min(100.0, score);
}

export function isConfirmedSpoofing(
  score: number,
  hasTeleport: boolean,
  hasVerticalSpike: boolean,
  hasAltitudeGlitch: boolean,
  confirmationThreshold: number = 75.0
): boolean {
  const meetsScore = score >= confirmationThreshold;
  const hasUnphysical = hasTeleport || hasVerticalSpike || hasAltitudeGlitch;
  return meetsScore && hasUnphysical;
}
