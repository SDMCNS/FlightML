/**
 * FlightML Aviation Economics & Passenger Liability (TypeScript).
 */

export function crcoWeightFactor(mtowTonnes: number): number {
  return Math.sqrt(Math.max(1.0, mtowTonnes) / 50.0);
}

export function crcoDistanceFactor(distanceKm: number): number {
  return Math.max(0.0, distanceKm) / 100.0;
}

export function crcoRouteChargeEur(
  distanceKm: number,
  mtowTonnes: number,
  unitRateEur: number
): number {
  return crcoDistanceFactor(distanceKm) * crcoWeightFactor(mtowTonnes) * Math.max(0.0, unitRateEur);
}

export function eu261StatutoryCompensationEur(
  greatCircleKm: number,
  arrivalDelayMin: number,
  isCancelled: boolean = false
): number {
  if (!isCancelled && arrivalDelayMin < 180.0) return 0.0;
  if (greatCircleKm <= 1500.0) return 250.0;
  if (greatCircleKm <= 3500.0) return 400.0;
  return 600.0;
}

export function passengerMissedConnectionProbability(
  inboundDelayMin: number,
  scheduledLayoverMin: number,
  minimumConnectTimeMin: number = 45.0,
  sigmoidSteepness: number = 0.18
): number {
  const bufferMin = scheduledLayoverMin - minimumConnectTimeMin;
  const delayExcessMin = inboundDelayMin - bufferMin;
  const clampedArg = Math.max(-25.0, Math.min(25.0, -sigmoidSteepness * delayExcessMin));
  return 1.0 / (1.0 + Math.exp(clampedArg));
}

export function passengerDisruptionLiabilityEur(
  paxCount: number,
  greatCircleKm: number,
  missedProbability: number,
  dutyOfCareHotelMealPaxEur: number = 160.0
): number {
  if (paxCount <= 0 || missedProbability <= 0) return 0.0;
  const eu261 = eu261StatutoryCompensationEur(greatCircleKm, 240.0);
  return paxCount * Math.min(1.0, Math.max(0.0, missedProbability)) * (eu261 + dutyOfCareHotelMealPaxEur);
}

export function delayCostByPhaseEur(
  gateDelayMin: number = 0,
  taxiDelayMin: number = 0,
  airborneDelayMin: number = 0
): number {
  return gateDelayMin * 45.0 + taxiDelayMin * 65.0 + airborneDelayMin * 95.0;
}
