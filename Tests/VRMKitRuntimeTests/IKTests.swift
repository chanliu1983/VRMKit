import Foundation
import Testing
import SIMDCompat
import VRMKit
@testable import VRMKitRuntime

@Suite
struct IKTests {
    private static func close(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ tolerance: Float = 1e-3) -> Bool {
        simd_distance(a, b) < tolerance
    }

    @Test
    func testTwoBoneReachesAPointInRangeAndKeepsItsBoneLengths() {
        let root = SIMD3<Float>(0, 0, 0)
        let solution = TwoBoneIK.solve(root: root, mid: SIMD3(1, 0, 0), end: SIMD3(2, 0, 0),
                                       target: SIMD3(1, 1, 0.5),
                                       poleDirection: SIMD3(0, 0, -1))
        #expect(Self.close(solution.end, SIMD3(1, 1, 0.5)))
        #expect(abs(simd_distance(root, solution.mid) - 1) < 1e-3)
        #expect(abs(simd_distance(solution.mid, solution.end) - 1) < 1e-3)
    }

    @Test
    func testTwoBoneBendsTowardThePole() {
        let solution = TwoBoneIK.solve(root: .zero, mid: SIMD3(1, 0, 0), end: SIMD3(2, 0, 0),
                                       target: SIMD3(1.2, 0, 0),
                                       poleDirection: SIMD3(0, 0, 1))
        #expect(solution.mid.z > 0.1)
    }

    @Test
    func testTwoBoneStretchesTowardAnUnreachableTargetWithoutLockingStraight() {
        let solution = TwoBoneIK.solve(root: .zero, mid: SIMD3(1, 0, 0), end: SIMD3(2, 0, 0),
                                       target: SIMD3(10, 0, 0),
                                       poleDirection: SIMD3(0, 1, 0))
        #expect(solution.end.x < 2)
        #expect(solution.end.x > 1.9)
        #expect(solution.mid.y > 0)
    }

    @Test
    func testTwoBoneStopsFoldingAtTheMinimumAngle() {
        let minimum: Float = 1.0
        let solution = TwoBoneIK.solve(root: .zero, mid: SIMD3(1, 0, 0), end: SIMD3(2, 0, 0),
                                       target: SIMD3(0.01, 0.01, 0),
                                       poleDirection: SIMD3(0, 1, 0),
                                       minInteriorAngle: minimum)
        let a = (SIMD3<Float>.zero - solution.mid)
        let b = (solution.end - solution.mid)
        let angle = acos(simd_clamp(simd_dot(simd_normalize(a), simd_normalize(b)), -1, 1))
        #expect(angle >= minimum - 1e-3)
    }

    @Test
    func testFABRIKReachesATargetAndKeepsBoneLengths() {
        let chain: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 2, 0), SIMD3(0, 3, 0)]
        let solved = FABRIK.solve(positions: chain, target: SIMD3(1.5, 1.5, 0.5))
        #expect(Self.close(solved[0], chain[0], 1e-5))
        #expect(Self.close(solved[3], SIMD3(1.5, 1.5, 0.5), 1e-2))
        for i in 1..<solved.count {
            #expect(abs(simd_distance(solved[i], solved[i - 1]) - 1) < 1e-3)
        }
    }

    @Test
    func testFABRIKStraightensTowardAnUnreachableTarget() {
        let chain: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 2, 0)]
        let solved = FABRIK.solve(positions: chain, target: SIMD3(10, 0, 0))
        #expect(Self.close(solved[2], SIMD3(2, 0, 0)))
    }

    /// An arm hanging down from a shoulder: root → upper arm → lower arm → hand, a unit apiece.
    private static func arm() -> GLTFPose {
        GLTFPose(children: [[1], [2], [3], []],
                 locals: [GLTFNodeTransform(),
                          GLTFNodeTransform(translation: SIMD3(0, -1, 0)),
                          GLTFNodeTransform(translation: SIMD3(0, -1, 0)),
                          GLTFNodeTransform(translation: SIMD3(0, -1, 0))])
    }

    @Test
    func testTheResolverMovesAHandOntoItsTargetThroughThePose() {
        let pose = Self.arm()
        let ik = IKResolver(pose: pose)
        let chain = ik.addChain(nodes: [1, 2, 3])
        chain.poleDirection = SIMD3(0, 0, -1)
        chain.smoothingTime = 0
        chain.targetPosition = SIMD3(1, -1, 0.5)
        ik.solve(deltaTime: 1 / 60)
        #expect(Self.close(pose.worldPosition(ofNode: 3), SIMD3(1, -1, 0.5)))
        // The root of the chain stays put.
        #expect(Self.close(pose.worldPosition(ofNode: 1), SIMD3(0, -1, 0)))
    }

    @Test
    func testAWeightOfZeroLeavesThePoseAlone() {
        let pose = Self.arm()
        let ik = IKResolver(pose: pose)
        let chain = ik.addChain(nodes: [1, 2, 3])
        chain.weight = 0
        chain.targetPosition = SIMD3(1, -1, 0)
        ik.solve(deltaTime: 1 / 60)
        #expect(Self.close(pose.worldPosition(ofNode: 3), SIMD3(0, -3, 0)))
    }

    @Test
    func testSmoothingApproachesTheTargetOverTime() {
        let pose = Self.arm()
        let ik = IKResolver(pose: pose)
        let chain = ik.addChain(nodes: [1, 2, 3])
        chain.smoothingTime = 0.1
        chain.targetPosition = SIMD3(1, -1, 0)
        ik.solve(deltaTime: 1 / 60)
        let first = pose.worldPosition(ofNode: 3)
        #expect(!Self.close(first, SIMD3(1, -1, 0), 0.05))
        for _ in 0..<120 { ik.solve(deltaTime: 1 / 60) }
        #expect(Self.close(pose.worldPosition(ofNode: 3), SIMD3(1, -1, 0), 0.02))
    }

    @Test
    func testALongerChainSolvesWithFABRIK() {
        let pose = GLTFPose(children: [[1], [2], [3], [4], []],
                            locals: [GLTFNodeTransform()] + (1..<5).map { _ in GLTFNodeTransform(translation: SIMD3(0, 1, 0)) })
        let ik = IKResolver(pose: pose)
        let chain = ik.addChain(nodes: [0, 1, 2, 3, 4])
        chain.smoothingTime = 0
        chain.targetPosition = SIMD3(2, 2, 1)
        ik.solve(deltaTime: 1 / 60)
        #expect(Self.close(pose.worldPosition(ofNode: 4), SIMD3(2, 2, 1), 2e-2))
    }
}
