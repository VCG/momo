"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.addRequiredJsonLogicOperations = addRequiredJsonLogicOperations;
exports.applyJsonLogic = applyJsonLogic;
exports.customJsonLogicOperations = void 0;
var _defineProperty2 = _interopRequireDefault(require("@babel/runtime/helpers/defineProperty"));
var _jsonLogicJs = _interopRequireDefault(require("json-logic-js"));
var _moment = _interopRequireDefault(require("moment"));
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { (0, _defineProperty2["default"])(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function applyJsonLogic(logic, data) {
  return _jsonLogicJs["default"].apply(logic, data);
}
function addJsonLogicOperation(name, op) {
  return _jsonLogicJs["default"].add_operation(name, op);
}
var customJsonLogicOperations = exports.customJsonLogicOperations = {
  CALL: function CALL(fn, ctx) {
    for (var _len = arguments.length, args = new Array(_len > 2 ? _len - 2 : 0), _key = 2; _key < _len; _key++) {
      args[_key - 2] = arguments[_key];
    }
    return fn.call.apply(fn, [ctx].concat(args));
  },
  JSX: function JSX(type, props) {
    return {
      type: type,
      props: props
    };
  },
  mergeObjects: function mergeObjects(obj1, obj2) {
    return _objectSpread(_objectSpread({}, obj1), obj2);
  },
  fromEntries: function fromEntries(entries) {
    return Object.fromEntries(entries);
  },
  strlen: function strlen(str) {
    return (str === null || str === void 0 ? void 0 : str.length) || 0;
  },
  regexTest: function regexTest(str, pattern, flags) {
    return (str === null || str === void 0 ? void 0 : str.match(new RegExp(pattern, flags))) != null;
  },
  now: function now() {
    return new Date();
  },
  date_add: function date_add(date, val, dim) {
    return (0, _moment["default"])(date).add(val, dim).toDate();
  },
  toLowerCase: function toLowerCase(str) {
    return str.toLowerCase();
  },
  toUpperCase: function toUpperCase(str) {
    return str.toUpperCase();
  }
};
function addRequiredJsonLogicOperations() {
  for (var k in customJsonLogicOperations) {
    addJsonLogicOperation(k, customJsonLogicOperations[k]);
  }
}