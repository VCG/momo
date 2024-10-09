"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.normalizeField = exports.getOperatorConfig = exports.getFuncConfig = exports.getFuncArgConfig = exports.getFieldWidgetConfig = exports.getFieldRawConfig = exports.getFieldConfig = exports.extendConfig = exports.createConfigMemo = void 0;
var _toConsumableArray2 = _interopRequireDefault(require("@babel/runtime/helpers/toConsumableArray"));
var _defineProperty2 = _interopRequireDefault(require("@babel/runtime/helpers/defineProperty"));
var _typeof2 = _interopRequireDefault(require("@babel/runtime/helpers/typeof"));
var _merge = _interopRequireDefault(require("lodash/merge"));
var _uuid = _interopRequireDefault(require("../utils/uuid"));
var _mergeWith = _interopRequireDefault(require("lodash/mergeWith"));
var _default = require("../config/default");
var _moment = _interopRequireDefault(require("moment"));
var _stuff = require("./stuff");
var _ruleUtils = require("./ruleUtils");
var _clone = _interopRequireDefault(require("clone"));
var _pick = _interopRequireDefault(require("lodash/pick"));
function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? ownKeys(Object(source), !0).forEach(function (key) { (0, _defineProperty2["default"])(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function _createForOfIteratorHelper(o, allowArrayLike) { var it = typeof Symbol !== "undefined" && o[Symbol.iterator] || o["@@iterator"]; if (!it) { if (Array.isArray(o) || (it = _unsupportedIterableToArray(o)) || allowArrayLike && o && typeof o.length === "number") { if (it) o = it; var i = 0; var F = function F() {}; return { s: F, n: function n() { if (i >= o.length) return { done: true }; return { done: false, value: o[i++] }; }, e: function e(_e) { throw _e; }, f: F }; } throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); } var normalCompletion = true, didErr = false, err; return { s: function s() { it = it.call(o); }, n: function n() { var step = it.next(); normalCompletion = step.done; return step; }, e: function e(_e2) { didErr = true; err = _e2; }, f: function f() { try { if (!normalCompletion && it["return"] != null) it["return"](); } finally { if (didErr) throw err; } } }; }
function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }
function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) arr2[i] = arr[i]; return arr2; }
var configKeys = ["conjunctions", "fields", "types", "operators", "widgets", "settings", "funcs"];
var pickConfig = function pickConfig(props) {
  return (0, _pick["default"])(props, configKeys);
};
var createConfigMemo = function createConfigMemo() {
  var configStore = new Map();
  var maxSize = 2; // current and prev
  var configId = 0;
  var extendAndStore = function extendAndStore(config) {
    var extendedConfig = extendConfig(config, ++configId);
    if (configStore.size + 1 > maxSize) {
      configStore["delete"](configStore.keys()[0]);
    }
    configStore.set(config, extendedConfig);
    return extendedConfig;
  };
  var findExtended = function findExtended(findConfig) {
    // strict find:
    // return configStore.get(findConfig) || configStore.values().find(ec => ec === findConfig);
    var _iterator = _createForOfIteratorHelper(configStore.keys()),
      _step;
    try {
      var _loop = function _loop() {
        var savedConfig = _step.value;
        var found = configKeys.map(function (k) {
          return savedConfig[k] === findConfig[k];
        }).filter(function (v) {
          return !v;
        }).length === 0;
        if (found) {
          return {
            v: configStore.get(savedConfig)
          };
        }
      };
      for (_iterator.s(); !(_step = _iterator.n()).done;) {
        var _ret = _loop();
        if ((0, _typeof2["default"])(_ret) === "object") return _ret.v;
      }
    } catch (err) {
      _iterator.e(err);
    } finally {
      _iterator.f();
    }
    var _iterator2 = _createForOfIteratorHelper(configStore.values()),
      _step2;
    try {
      var _loop2 = function _loop2() {
        var extendedConfig = _step2.value;
        var found = configKeys.map(function (k) {
          return extendedConfig[k] === findConfig[k];
        }).filter(function (v) {
          return !v;
        }).length === 0;
        if (found) {
          return {
            v: extendedConfig
          };
        }
      };
      for (_iterator2.s(); !(_step2 = _iterator2.n()).done;) {
        var _ret2 = _loop2();
        if ((0, _typeof2["default"])(_ret2) === "object") return _ret2.v;
      }
    } catch (err) {
      _iterator2.e(err);
    } finally {
      _iterator2.f();
    }
    return null;
  };
  var findOrExtend = function findOrExtend(config) {
    return findExtended(config) || extendAndStore(config);
  };
  return function (props) {
    return findOrExtend(pickConfig(props));
  };
};
exports.createConfigMemo = createConfigMemo;
var extendConfig = function extendConfig(config, configId) {
  //operators, defaultOperator - merge
  //widgetProps (including valueLabel, valuePlaceholder, hideOperator, operatorInlineLabel) - concrete by widget

  if (config.__configId) {
    return config;
  }
  config = _objectSpread({}, config);
  config.settings = (0, _merge["default"])({}, _default.settings, config.settings);
  config._fieldsCntByType = {};
  config._funcsCntByType = {};
  config.types = (0, _clone["default"])(config.types);
  _extendTypesConfig(config.types, config);
  config.fields = (0, _clone["default"])(config.fields);
  config.__fieldNames = {};
  _extendFieldsConfig(config.fields, config);
  config.funcs = (0, _clone["default"])(config.funcs);
  _extendFuncArgsConfig(config.funcs, config);
  _moment["default"].locale(config.settings.locale.moment);
  Object.defineProperty(config, "__configId", {
    enumerable: false,
    writable: false,
    value: configId || (0, _uuid["default"])()
  });
  return config;
};
exports.extendConfig = extendConfig;
function _extendTypesConfig(typesConfig, config) {
  for (var type in typesConfig) {
    var typeConfig = typesConfig[type];
    _extendTypeConfig(type, typeConfig, config);
  }
}
function _extendTypeConfig(type, typeConfig, config) {
  var operators = null,
    defaultOperator = null;
  typeConfig.mainWidget = typeConfig.mainWidget || Object.keys(typeConfig.widgets).filter(function (w) {
    return w != "field" && w != "func";
  })[0];
  for (var widget in typeConfig.widgets) {
    var typeWidgetConfig = typeConfig.widgets[widget];
    if (typeWidgetConfig.operators) {
      var typeWidgetOperators = typeWidgetConfig.operators;
      if (typeConfig.excludeOperators) {
        typeWidgetOperators = typeWidgetOperators.filter(function (op) {
          return !typeConfig.excludeOperators.includes(op);
        });
      }
      operators = (0, _stuff.mergeArraysSmart)(operators, typeWidgetOperators);
    }
    if (typeWidgetConfig.defaultOperator) defaultOperator = typeWidgetConfig.defaultOperator;
    if (widget == typeConfig.mainWidget) {
      typeWidgetConfig = (0, _merge["default"])({}, {
        widgetProps: typeConfig.mainWidgetProps || {}
      }, typeWidgetConfig);
    }
    typeConfig.widgets[widget] = typeWidgetConfig;
  }
  if (!typeConfig.valueSources) typeConfig.valueSources = Object.keys(config.settings.valueSourcesInfo);
  var _iterator3 = _createForOfIteratorHelper(typeConfig.valueSources),
    _step3;
  try {
    for (_iterator3.s(); !(_step3 = _iterator3.n()).done;) {
      var valueSrc = _step3.value;
      if (valueSrc != "value" && !typeConfig.widgets[valueSrc]) {
        typeConfig.widgets[valueSrc] = {};
      }
    }
  } catch (err) {
    _iterator3.e(err);
  } finally {
    _iterator3.f();
  }
  if (!typeConfig.operators && operators) typeConfig.operators = Array.from(new Set(operators)); //unique
  if (!typeConfig.defaultOperator && defaultOperator) typeConfig.defaultOperator = defaultOperator;
}
function _extendFieldsConfig(subconfig, config) {
  var path = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : [];
  for (var field in subconfig) {
    _extendFieldConfig(subconfig[field], config, [].concat((0, _toConsumableArray2["default"])(path), [field]));
    if (subconfig[field].subfields) {
      _extendFieldsConfig(subconfig[field].subfields, config, [].concat((0, _toConsumableArray2["default"])(path), [field]));
    }
  }
}
function _extendFuncArgsConfig(subconfig, config) {
  if (!subconfig) return;
  for (var funcKey in subconfig) {
    var funcDef = subconfig[funcKey];
    if (funcDef.returnType) {
      if (!config._funcsCntByType[funcDef.returnType]) config._funcsCntByType[funcDef.returnType] = 0;
      config._funcsCntByType[funcDef.returnType]++;
    }
    for (var argKey in funcDef.args) {
      _extendFieldConfig(funcDef.args[argKey], config, null, true);
    }

    // isOptional can be only in the end
    if (funcDef.args) {
      var argKeys = Object.keys(funcDef.args);
      var tmpIsOptional = true;
      var _iterator4 = _createForOfIteratorHelper(argKeys.reverse()),
        _step4;
      try {
        for (_iterator4.s(); !(_step4 = _iterator4.n()).done;) {
          var _argKey = _step4.value;
          var argDef = funcDef.args[_argKey];
          if (!tmpIsOptional && argDef.isOptional) {
            delete argDef.isOptional;
          }
          if (!argDef.isOptional) tmpIsOptional = false;
        }
      } catch (err) {
        _iterator4.e(err);
      } finally {
        _iterator4.f();
      }
    }
    if (funcDef.subfields) {
      _extendFuncArgsConfig(funcDef.subfields, config);
    }
  }
}
function _extendFieldConfig(fieldConfig, config) {
  var path = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : null;
  var isFuncArg = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : false;
  var operators = null,
    defaultOperator = null;
  var typeConfig = config.types[fieldConfig.type];
  var excludeOperatorsForField = fieldConfig.excludeOperators || [];
  if (fieldConfig.type != "!struct" && fieldConfig.type != "!group") {
    var keysToPutInFieldSettings = ["listValues", "allowCustomValues", "validateValue"];
    if (!fieldConfig.fieldSettings) fieldConfig.fieldSettings = {};
    for (var _i = 0, _keysToPutInFieldSett = keysToPutInFieldSettings; _i < _keysToPutInFieldSett.length; _i++) {
      var k = _keysToPutInFieldSett[_i];
      if (fieldConfig[k]) {
        fieldConfig.fieldSettings[k] = fieldConfig[k];
        delete fieldConfig[k];
      }
    }
    if (fieldConfig.fieldSettings.listValues) {
      fieldConfig.fieldSettings.listValues = (0, _stuff.normalizeListValues)(fieldConfig.fieldSettings.listValues, fieldConfig.type, fieldConfig.fieldSettings);
    }
    if (!typeConfig) {
      //console.warn(`No type config for ${fieldConfig.type}`);
      fieldConfig.disabled = true;
      return;
    }
    if (!isFuncArg) {
      if (!config._fieldsCntByType[fieldConfig.type]) config._fieldsCntByType[fieldConfig.type] = 0;
      config._fieldsCntByType[fieldConfig.type]++;
    }
    if (!fieldConfig.widgets) fieldConfig.widgets = {};
    if (isFuncArg) fieldConfig._isFuncArg = true;
    fieldConfig.mainWidget = fieldConfig.mainWidget || typeConfig.mainWidget;
    fieldConfig.valueSources = fieldConfig.valueSources || typeConfig.valueSources;
    var excludeOperatorsForType = typeConfig.excludeOperators || [];
    var _loop3 = function _loop3() {
      var fieldWidgetConfig = fieldConfig.widgets[widget] || {};
      var typeWidgetConfig = typeConfig.widgets[widget] || {};
      if (!isFuncArg) {
        //todo: why I've excluded isFuncArg ?
        var excludeOperators = [].concat((0, _toConsumableArray2["default"])(excludeOperatorsForField), (0, _toConsumableArray2["default"])(excludeOperatorsForType));
        var shouldIncludeOperators = fieldConfig.preferWidgets && (widget == "field" || fieldConfig.preferWidgets.includes(widget)) || excludeOperators.length > 0;
        if (fieldWidgetConfig.operators) {
          var addOperators = fieldWidgetConfig.operators.filter(function (o) {
            return !excludeOperators.includes(o);
          });
          operators = [].concat((0, _toConsumableArray2["default"])(operators || []), (0, _toConsumableArray2["default"])(addOperators));
        } else if (shouldIncludeOperators && typeWidgetConfig.operators) {
          var _addOperators = typeWidgetConfig.operators.filter(function (o) {
            return !excludeOperators.includes(o);
          });
          operators = [].concat((0, _toConsumableArray2["default"])(operators || []), (0, _toConsumableArray2["default"])(_addOperators));
        }
        if (fieldWidgetConfig.defaultOperator) defaultOperator = fieldWidgetConfig.defaultOperator;
      }
      if (widget == fieldConfig.mainWidget) {
        fieldWidgetConfig = (0, _merge["default"])({}, {
          widgetProps: fieldConfig.mainWidgetProps || {}
        }, fieldWidgetConfig);
      }
      fieldConfig.widgets[widget] = fieldWidgetConfig;
    };
    for (var widget in typeConfig.widgets) {
      _loop3();
    }
    if (!isFuncArg) {
      if (!fieldConfig.operators && operators) fieldConfig.operators = Array.from(new Set(operators));
      if (!fieldConfig.defaultOperator && defaultOperator) fieldConfig.defaultOperator = defaultOperator;
    }
  }
  var computedFieldName = computeFieldName(config, path);
  if (computedFieldName) {
    fieldConfig.fieldName = computedFieldName;
  }
  if (path && fieldConfig.fieldName) {
    config.__fieldNames[fieldConfig.fieldName] = path;
  }
}
var getFieldRawConfig = function getFieldRawConfig(config, field) {
  var fieldsKey = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : "fields";
  var subfieldsKey = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : "subfields";
  if (!field) return null;
  if (field == "!case_value") {
    return {
      type: "case_value",
      mainWidget: "case_value",
      widgets: {
        "case_value": config.widgets["case_value"]
      }
    };
  }
  var fieldSeparator = config.settings.fieldSeparator;
  //field = normalizeField(config, field);
  var parts = Array.isArray(field) ? field : field.split(fieldSeparator);
  var targetFields = config[fieldsKey];
  if (!targetFields) return null;
  var fields = targetFields;
  var fieldConfig = null;
  var path = [];
  for (var i = 0; i < parts.length; i++) {
    var part = parts[i];
    path.push(part);
    var pathKey = path.join(fieldSeparator);
    fieldConfig = fields[pathKey];
    if (i < parts.length - 1) {
      if (fieldConfig && fieldConfig[subfieldsKey]) {
        fields = fieldConfig[subfieldsKey];
        path = [];
      } else {
        fieldConfig = null;
      }
    }
  }
  return fieldConfig;
};
exports.getFieldRawConfig = getFieldRawConfig;
var computeFieldName = function computeFieldName(config, path) {
  if (!path) return null;
  var fieldSeparator = config.settings.fieldSeparator;
  var l = (0, _toConsumableArray2["default"])(path),
    r = [],
    f,
    fConfig;
  while ((f = l.pop()) !== undefined && l.length > 0) {
    r.unshift(f);
    fConfig = getFieldRawConfig(config, l);
    if (fConfig.fieldName) {
      return [fConfig.fieldName].concat(r).join(fieldSeparator);
    }
  }
  return null;
};
var normalizeField = function normalizeField(config, field) {
  var fieldSeparator = config.settings.fieldSeparator;
  var fieldStr = Array.isArray(field) ? field.join(fieldSeparator) : field;
  if (config.__fieldNames[fieldStr]) {
    return config.__fieldNames[fieldStr].join(fieldSeparator);
  }
  return fieldStr;
};
exports.normalizeField = normalizeField;
var getFuncConfig = function getFuncConfig(config, func) {
  if (!func) return null;
  var funcConfig = getFieldRawConfig(config, func, "funcs", "subfields");
  if (!funcConfig) return null; //throw new Error("Can't find func " + func + ", please check your config");
  return funcConfig;
};
exports.getFuncConfig = getFuncConfig;
var getFuncArgConfig = function getFuncArgConfig(config, funcKey, argKey) {
  var funcConfig = getFuncConfig(config, funcKey);
  if (!funcConfig) return null; //throw new Error(`Can't find func ${funcKey}, please check your config`);
  var argConfig = funcConfig.args && funcConfig.args[argKey] || null;
  if (!argConfig) return null; //throw new Error(`Can't find arg ${argKey} for func ${funcKey}, please check your config`);

  //merge, but don't merge operators (rewrite instead)
  var typeConfig = config.types[argConfig.type] || {};
  var ret = (0, _mergeWith["default"])({}, typeConfig, argConfig || {}, function (objValue, srcValue, _key, _object, _source, _stack) {
    if (Array.isArray(objValue)) {
      return srcValue;
    }
  });
  return ret;
};
exports.getFuncArgConfig = getFuncArgConfig;
var getFieldConfig = function getFieldConfig(config, field) {
  if (!field) return null;
  if ((0, _typeof2["default"])(field) == "object" && !field.func && !!field.type) return field;
  if ((0, _typeof2["default"])(field) == "object" && field.func && field.arg) return getFuncArgConfig(config, field.func, field.arg);
  var fieldConfig = getFieldRawConfig(config, field);
  if (!fieldConfig) return null; //throw new Error("Can't find field " + field + ", please check your config");

  //merge, but don't merge operators (rewrite instead)
  var typeConfig = config.types[fieldConfig.type] || {};
  var ret = (0, _mergeWith["default"])({}, typeConfig, fieldConfig || {}, function (objValue, srcValue, _key, _object, _source, _stack) {
    if (Array.isArray(objValue)) {
      return srcValue;
    }
  });
  return ret;
};
exports.getFieldConfig = getFieldConfig;
var getOperatorConfig = function getOperatorConfig(config, operator) {
  var field = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : null;
  if (!operator) return null;
  var opConfig = config.operators[operator];
  if (field) {
    var fieldConfig = getFieldConfig(config, field);
    var widget = (0, _ruleUtils.getWidgetForFieldOp)(config, field, operator);
    var widgetConfig = config.widgets[widget] || {};
    var fieldWidgetConfig = (fieldConfig && fieldConfig.widgets ? fieldConfig.widgets[widget] : {}) || {};
    var widgetOpProps = (widgetConfig.opProps || {})[operator];
    var fieldWidgetOpProps = (fieldWidgetConfig.opProps || {})[operator];
    var mergedOpConfig = (0, _merge["default"])({}, opConfig, widgetOpProps, fieldWidgetOpProps);
    return mergedOpConfig;
  } else {
    return opConfig;
  }
};
exports.getOperatorConfig = getOperatorConfig;
var getFieldWidgetConfig = function getFieldWidgetConfig(config, field, operator) {
  var widget = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : null;
  var valueSrc = arguments.length > 4 && arguments[4] !== undefined ? arguments[4] : null;
  if (!field) return null;
  if (!(operator || widget) && valueSrc != "const" && field != "!case_value") return null;
  var fieldConfig = getFieldConfig(config, field);
  if (!widget) widget = (0, _ruleUtils.getWidgetForFieldOp)(config, field, operator, valueSrc);
  var widgetConfig = config.widgets[widget] || {};
  var fieldWidgetConfig = (fieldConfig && fieldConfig.widgets ? fieldConfig.widgets[widget] : {}) || {};
  var fieldWidgetProps = fieldWidgetConfig.widgetProps || {};
  var valueFieldSettings = (valueSrc == "value" || !valueSrc) && fieldConfig && fieldConfig.fieldSettings || {}; // useful to take 'validateValue'
  var mergedConfig = (0, _merge["default"])({}, widgetConfig, fieldWidgetProps, valueFieldSettings);
  return mergedConfig;
};
exports.getFieldWidgetConfig = getFieldWidgetConfig;