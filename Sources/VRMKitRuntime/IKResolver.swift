#if canImport(FoundationEssentials)
import FoundationEssentials
#else
import Foundation
#endif
import SIMDCompat
import VRMKit

/// A chain of bones that reaches for a goal, such as an arm reaching for a point.
public final class IKChain {
    /// The bones, root first, as nodes of the pose. Two bones plus the end make three nodes
    /// and solve analytically; longer chains solve with FABRIK.
    public let nodes: [Int]

    /// Where the end of the chain should be, in world space.
    public var targetPosition: SIMD3<Float>

    /// The rotation the end should face in world space, or nil to leave it as animated.
    public var targetRotation: simd_quatf?

    /// How much the solve replaces the animation: 0 leaves the pose alone, 1 reaches the goal.
    public var weight: Float = 1

    /// Which way the middle joint bends, a world direction from the root, such as forward for
    /// a knee. Nil keeps the side the joint is bent to.
    public var poleDirection: SIMD3<Float>?

    /// The tightest the middle joint folds, in radians, for a chain of three nodes.
    public var minInteriorAngle: Float = 0.2

    /// How far the middle joint opens, in radians. Below pi it stops the limb locking straight.
    public var maxInteriorAngle: Float = .pi - 0.02

    /// The time in seconds the goal takes to cover most of the way to a new target. 0 snaps.
    public var smoothingTime: Float = 0.08

    /// The goal as smoothed, which the solve reaches for.
    package private(set) var smoothedTarget: SIMD3<Float>
    private var hasSmoothedTarget = false

    public init(nodes: [Int], targetPosition: SIMD3<Float> = .zero) {
        precondition(nodes.count >= 3, "a chain needs at least a root, a middle and an end")
        self.nodes = nodes
        self.targetPosition = targetPosition
        self.smoothedTarget = targetPosition
    }

    /// Jumps the smoothed goal onto the target, as when an object is teleported.
    public func snapToTarget() {
        smoothedTarget = targetPosition
        hasSmoothedTarget = true
    }

    fileprivate func smooth(deltaTime: Float) {
        guard hasSmoothedTarget, smoothingTime > 1e-5 else {
            snapToTarget()
            return
        }
        // Frame rate independent: the same fraction of the gap closes per second at any rate.
        let blend = 1 - exp(-max(0, deltaTime) / smoothingTime)
        smoothedTarget += (targetPosition - smoothedTarget) * blend
    }
}

/// Solves IK chains on a model's pose, each frame.
///
/// Make one from a model's pose, add chains, move their targets, and call ``solve(deltaTime:)``
/// from an update handler at ``VRMUpdateStage/afterAnimation`` so the solve lands on top of
/// the animation and the constraints and spring bones follow it:
///
/// ```swift
/// let ik = IKResolver(pose: pose)
/// let hand = ik.addLimb(upper: .rightUpperArm, lower: .rightLowerArm, end: .rightHand, humanoid: model.humanoid)
/// model.setUpdateHandler({ dt in
///     hand?.targetPosition = reachPoint
///     ik.solve(deltaTime: Float(dt))
/// }, at: .afterAnimation)
/// ```
public final class IKResolver {
    public let pose: GLTFPose
    public private(set) var chains: [IKChain] = []

    public init(pose: GLTFPose) {
        self.pose = pose
    }

    @discardableResult
    public func addChain(nodes: [Int], targetPosition: SIMD3<Float>? = nil) -> IKChain {
        let chain = IKChain(nodes: nodes, targetPosition: targetPosition ?? pose.worldPosition(ofNode: nodes[nodes.count - 1]))
        chain.snapToTarget()
        chains.append(chain)
        return chain
    }

    /// A two-bone chain of humanoid bones, nil when the model lacks one of them.
    @discardableResult
    public func addLimb<Node>(upper: HumanoidBone, lower: HumanoidBone, end: HumanoidBone,
                              humanoid: Humanoid<Node>) -> IKChain? {
        guard let upperNode = humanoid.nodeIndex(for: upper),
              let lowerNode = humanoid.nodeIndex(for: lower),
              let endNode = humanoid.nodeIndex(for: end) else { return nil }
        return addChain(nodes: [upperNode, lowerNode, endNode])
    }

    /// A chain of any length from a humanoid bone to another, along the bones between them in
    /// the order the pose's hierarchy has them, nil when they are not on one line of descent.
    @discardableResult
    public func addChain<Node>(from start: HumanoidBone, to end: HumanoidBone,
                               humanoid: Humanoid<Node>) -> IKChain? {
        guard let startNode = humanoid.nodeIndex(for: start),
              let endNode = humanoid.nodeIndex(for: end) else { return nil }
        var path = [endNode]
        while let last = path.last, last != startNode {
            guard let parent = pose.parent(ofNode: last) else { return nil }
            path.append(parent)
        }
        path.reverse()
        guard path.count >= 3 else { return nil }
        return addChain(nodes: path)
    }

    public func removeChain(_ chain: IKChain) {
        chains.removeAll { $0 === chain }
    }

    /// Solves every chain with weight, leaving the rest of the pose as it is.
    public func solve(deltaTime: Float) {
        for chain in chains {
            chain.smooth(deltaTime: deltaTime)
            guard chain.weight > 1e-4 else { continue }
            solve(chain)
        }
    }

    private func solve(_ chain: IKChain) {
        let nodes = chain.nodes
        let positions = nodes.map { pose.worldPosition(ofNode: $0) }
        let end = positions[positions.count - 1]
        let weight = simd_clamp(chain.weight, 0, 1)
        // Blending the goal blends the solve, so a weight between 0 and 1 reaches part way.
        let target = end + (chain.smoothedTarget - end) * weight

        let solved: [SIMD3<Float>]
        if nodes.count == 3 {
            let solution = TwoBoneIK.solve(root: positions[0], mid: positions[1], end: positions[2],
                                           target: target,
                                           poleDirection: chain.poleDirection,
                                           minInteriorAngle: chain.minInteriorAngle,
                                           maxInteriorAngle: chain.maxInteriorAngle)
            solved = [positions[0], solution.mid, solution.end]
        } else {
            solved = FABRIK.solve(positions: positions, target: target)
        }

        // Turn each bone from root to tip so it points at the next solved joint. A bone's turn
        // moves everything below it, so the next bone's direction is read after.
        for i in 0..<(nodes.count - 1) {
            let current = pose.worldPosition(ofNode: nodes[i + 1]) - pose.worldPosition(ofNode: nodes[i])
            let wanted = solved[i + 1] - solved[i]
            let turn = ikRotation(from: current, to: wanted)
            pose.setWorldRotation(turn * pose.worldRotation(ofNode: nodes[i]), ofNode: nodes[i])
        }

        if let rotation = chain.targetRotation {
            let tip = nodes[nodes.count - 1]
            let blended = simd_slerp(pose.worldRotation(ofNode: tip), rotation, weight)
            pose.setWorldRotation(blended, ofNode: tip)
        }
    }
}
