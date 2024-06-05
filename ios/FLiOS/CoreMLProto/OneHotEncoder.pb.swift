// DO NOT EDIT.
// swift-format-ignore-file
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: OneHotEncoder.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

// Copyright (c) 2017, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in LICENSE.txt or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that you are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
  typealias Version = _2
}

///*
/// Transforms a categorical feature into an array. The array will be all
/// zeros expect a single entry of one.
///
/// Each categorical value will map to an index, this mapping is given by
/// either the ``stringCategories`` parameter or the ``int64Categories``
/// parameter.
struct CoreML_Specification_OneHotEncoder {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  ///*
  /// Mapping to be used for the encoding. The position of the category in
  /// the below vector determines where the single one entry will be in the
  /// output.
  var categoryType: CoreML_Specification_OneHotEncoder.OneOf_CategoryType? = nil

  var stringCategories: CoreML_Specification_StringVector {
    get {
      if case .stringCategories(let v)? = categoryType {return v}
      return CoreML_Specification_StringVector()
    }
    set {categoryType = .stringCategories(newValue)}
  }

  var int64Categories: CoreML_Specification_Int64Vector {
    get {
      if case .int64Categories(let v)? = categoryType {return v}
      return CoreML_Specification_Int64Vector()
    }
    set {categoryType = .int64Categories(newValue)}
  }

  /// Output can be a dictionary with only one entry, instead of an array.
  var outputSparse: Bool = false

  var handleUnknown: CoreML_Specification_OneHotEncoder.HandleUnknown = .errorOnUnknown

  var unknownFields = SwiftProtobuf.UnknownStorage()

  ///*
  /// Mapping to be used for the encoding. The position of the category in
  /// the below vector determines where the single one entry will be in the
  /// output.
  enum OneOf_CategoryType: Equatable {
    case stringCategories(CoreML_Specification_StringVector)
    case int64Categories(CoreML_Specification_Int64Vector)

  #if !swift(>=4.1)
    static func ==(lhs: CoreML_Specification_OneHotEncoder.OneOf_CategoryType, rhs: CoreML_Specification_OneHotEncoder.OneOf_CategoryType) -> Bool {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch (lhs, rhs) {
      case (.stringCategories, .stringCategories): return {
        guard case .stringCategories(let l) = lhs, case .stringCategories(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      case (.int64Categories, .int64Categories): return {
        guard case .int64Categories(let l) = lhs, case .int64Categories(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      default: return false
      }
    }
  #endif
  }

  enum HandleUnknown: SwiftProtobuf.Enum {
    typealias RawValue = Int
    case errorOnUnknown // = 0

    /// Output will be all zeros for unknown values.
    case ignoreUnknown // = 1
    case UNRECOGNIZED(Int)

    init() {
      self = .errorOnUnknown
    }

    init?(rawValue: Int) {
      switch rawValue {
      case 0: self = .errorOnUnknown
      case 1: self = .ignoreUnknown
      default: self = .UNRECOGNIZED(rawValue)
      }
    }

    var rawValue: Int {
      switch self {
      case .errorOnUnknown: return 0
      case .ignoreUnknown: return 1
      case .UNRECOGNIZED(let i): return i
      }
    }

  }

  init() {}
}

#if swift(>=4.2)

extension CoreML_Specification_OneHotEncoder.HandleUnknown: CaseIterable {
  // The compiler won't synthesize support with the UNRECOGNIZED case.
  static var allCases: [CoreML_Specification_OneHotEncoder.HandleUnknown] = [
    .errorOnUnknown,
    .ignoreUnknown,
  ]
}

#endif  // swift(>=4.2)

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification"

extension CoreML_Specification_OneHotEncoder: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".OneHotEncoder"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "stringCategories"),
    2: .same(proto: "int64Categories"),
    10: .same(proto: "outputSparse"),
    11: .same(proto: "handleUnknown"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try {
        var v: CoreML_Specification_StringVector?
        var hadOneofValue = false
        if let current = self.categoryType {
          hadOneofValue = true
          if case .stringCategories(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.categoryType = .stringCategories(v)
        }
      }()
      case 2: try {
        var v: CoreML_Specification_Int64Vector?
        var hadOneofValue = false
        if let current = self.categoryType {
          hadOneofValue = true
          if case .int64Categories(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.categoryType = .int64Categories(v)
        }
      }()
      case 10: try { try decoder.decodeSingularBoolField(value: &self.outputSparse) }()
      case 11: try { try decoder.decodeSingularEnumField(value: &self.handleUnknown) }()
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    // The use of inline closures is to circumvent an issue where the compiler
    // allocates stack space for every if/case branch local when no optimizations
    // are enabled. https://github.com/apple/swift-protobuf/issues/1034 and
    // https://github.com/apple/swift-protobuf/issues/1182
    switch self.categoryType {
    case .stringCategories?: try {
      guard case .stringCategories(let v)? = self.categoryType else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
    }()
    case .int64Categories?: try {
      guard case .int64Categories(let v)? = self.categoryType else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
    }()
    case nil: break
    }
    if self.outputSparse != false {
      try visitor.visitSingularBoolField(value: self.outputSparse, fieldNumber: 10)
    }
    if self.handleUnknown != .errorOnUnknown {
      try visitor.visitSingularEnumField(value: self.handleUnknown, fieldNumber: 11)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_OneHotEncoder, rhs: CoreML_Specification_OneHotEncoder) -> Bool {
    if lhs.categoryType != rhs.categoryType {return false}
    if lhs.outputSparse != rhs.outputSparse {return false}
    if lhs.handleUnknown != rhs.handleUnknown {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_OneHotEncoder.HandleUnknown: SwiftProtobuf._ProtoNameProviding {
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    0: .same(proto: "ErrorOnUnknown"),
    1: .same(proto: "IgnoreUnknown"),
  ]
}
