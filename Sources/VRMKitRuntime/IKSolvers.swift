import SIMDCompat
import VRMKit

/// The rotation that turns `from` onto `to`, both directions in the same space.
package func ikRotation(from: SIMD3<Float>, to: SIMD3<Float>) -> simd_quatf {
    let fromLength = simd_length(from)
    let toLength = simd_length(to)
    guard fromLength > 1e-8, toLength > 1e-8 else { return quat_identity_float }
    let a = from / fromLength
    let b = to / toLength
    let dot = simd_clamp(simd_dot(a, b), -1, 1)
    if dot > 1 - 1e-7 { return quat_identity_float }
    if dot < -1 + 1e-7 {
        // Opposite directions: any axis perpendicular to `a` turns it half way round.
        var axis = simd_cross(a, SIMD3<Float>(1, 0, 0))
        if simd_length_squared(axis) < 1e-6 { axis = simd_cross(a, SIMD3<Float>(0, 1, 0)) }
        return simd_quatf(angle: .pi, axis: simd_normalize(axis))
    }
    let axis = simd_cross(a, b)
    return simd_quatf(angle: acos(dot), axis: simd_normalize(axis))
}

/// Analytic IK for a chain of two bones, such as an arm or a leg.
public enum TwoBoneIK {
    /// Where the middle joint and the end of the chain end up.
    public struct Solution: Equatable, Sendable {
        public var mid: SIMD3<Float>
        public var end: SIMD3<Float>
    }

    /// Puts the end of the `root` → `mid` → `end` chain on `target`, or as near it as the
    /// bones and the joint limits reach.
    ///
    /// - Parameters:
    ///   - poleDirection: Which way the middle joint points, a direction from the root,
    ///     such as forward for a knee. Without one the joint keeps the side it is bent to now.
    ///   - minInteriorAngle: The tightest the middle joint folds, in radians. An elbow
    ///     folded onto itself is 0, an arm held straight is pi.
    ///   - maxInteriorAngle: How far the middle joint opens, in radians. Below pi it keeps
    ///     the limb from locking straight.
    public static func solve(root: SIMD3<Float>,
                             mid: SIMD3<Float>,
                             end: SIMD3<Float>,
                             target: SIMD3<Float>,
                             poleDirection: SIMD3<Float>? = nil,
                             minInteriorAngle: Float = 0.2,
                             maxInteriorAngle: Float = .pi - 0.02) -> Solution {
        let upper = simd_length(mid - root)
        let lower = simd_length(end - mid)
        let toTarget = target - root
        let targetDistance = simd_length(toTarget)
        guard upper > 1e-6, lower > 1e-6, targetDistance > 1e-6 else {
            return Solution(mid: mid, end: end)
        }
        let direction = toTarget / targetDistance

        // The joint limits bound how near and how far the end can reach.
        let minInterior = simd_clamp(minInteriorAngle, 0, .pi)
        let maxInterior = simd_clamp(maxInteriorAngle, minInterior, .pi)
        let nearest = reach(upper: upper, lower: lower, interiorAngle: minInterior)
        let farthest = reach(upper: upper, lower: lower, interiorAngle: maxInterior)
        let distance = min(max(targetDistance, nearest), farthest)

        // Which side the joint bends to: the pole, or else where it points now.
        var bend = poleDirection ?? (mid - root)
        bend -= direction * simd_dot(bend, direction)
        if simd_length_squared(bend) < 1e-10 {
            bend = (mid - root) - direction * simd_dot(mid - root, direction)
        }
        if simd_length_squared(bend) < 1e-10 {
            // A limb held dead straight at the target: any side will do.
            bend = simd_cross(direction, abs(direction.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0))
        }
        bend = simd_normalize(bend)

        // The law of cosines gives the angle at the root between the target and the upper bone.
        let cosRoot = simd_clamp((upper * upper + distance * distance - lower * lower) / (2 * upper * distance), -1, 1)
        let sinRoot = (1 - cosRoot * cosRoot).squareRoot()
        let newMid = root + direction * (upper * cosRoot) + bend * (upper * sinRoot)
        let newEnd = root + direction * distance
        return Solution(mid: newMid, end: newEnd)
    }

    /// The distance from root to end when the middle joint is folded to `interiorAngle`.
    static func reach(upper: Float, lower: Float, interiorAngle: Float) -> Float {
        (upper * upper + lower * lower - 2 * upper * lower * cos(interiorAngle)).squareRoot()
    }
}

/// FABRIK for chains of any length, such as a spine or a tail.
public enum FABRIK {
    /// Moves `positions`, root first, so that the last reaches `target`, or as near as the
    /// bone lengths allow. The root stays where it is and each bone keeps its length.
    public static func solve(positions: [SIMD3<Float>],
                             target: SIMD3<Float>,
                             iterations: Int = 12,
                             tolerance: Float = 1e-4) -> [SIMD3<Float>] {
        guard positions.count >= 2 else { return positions }
        var p = positions
        let root = p[0]
        let lengths = (1..<p.count).map { simd_length(p[$0] - p[$0 - 1]) }
        let total = lengths.reduce(0, +)

        if simd_length(target - root) >= total {
            // Out of reach: stretch the chain straight at the target.
            let direction = simd_length(target - root) > 1e-8 ? simd_normalize(target - root) : SIMD3<Float>(0, 1, 0)
            var at = root
            for i in 1..<p.count {
                at += direction * lengths[i - 1]
                p[i] = at
            }
            return p
        }

        for _ in 0..<max(1, iterations) {
            if simd_length(p[p.count - 1] - target) < tolerance { break }
            // Backward: pin the end to the target and pull the chain after it.
            p[p.count - 1] = target
            for i in stride(from: p.count - 2, through: 0, by: -1) {
                p[i] = place(p[i], from: p[i + 1], length: lengths[i])
            }
            // Forward: pin the root back and push the chain out again.
            p[0] = root
            for i in 1..<p.count {
                p[i] = place(p[i], from: p[i - 1], length: lengths[i - 1])
            }
        }
        return p
    }

    /// `point` moved along its line to `anchor`, to `length` away from it.
    private static func place(_ point: SIMD3<Float>, from anchor: SIMD3<Float>, length: Float) -> SIMD3<Float> {
        let offset = point - anchor
        let distance = simd_length(offset)
        guard distance > 1e-8 else { return anchor + SIMD3<Float>(0, length, 0) }
        return anchor + offset * (length / distance)
    }
}
