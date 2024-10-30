"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.UPPER = exports.RELATIVE_DATETIME = exports.NOW = exports.LOWER = exports.LINEAR_REGRESSION = void 0;
var _slicedToArray2 = _interopRequireDefault(require("@babel/runtime/helpers/slicedToArray"));
//import { customJsonLogicOperations } from "../utils/jsonLogic";

// Tip: search for `customJsonLogicOperations` in codebase to see custom JL funcs we use in `jsonLogicCustomOps`

var NOW = exports.NOW = {
  label: "Now",
  returnType: "datetime",
  jsonLogic: "now",
  jsonLogicCustomOps: {
    now: {}
  },
  //spelFunc: "new java.util.Date()",
  spelFunc: "T(java.time.LocalDateTime).now()",
  sqlFormatFunc: function sqlFormatFunc() {
    return "NOW()";
  },
  mongoFormatFunc: function mongoFormatFunc() {
    return new Date();
  },
  formatFunc: function formatFunc() {
    return "NOW";
  }
};
var RELATIVE_DATETIME = exports.RELATIVE_DATETIME = {
  label: "Relative",
  returnType: "datetime",
  renderBrackets: ["", ""],
  renderSeps: ["", "", ""],
  spelFormatFunc: function spelFormatFunc(_ref) {
    var date = _ref.date,
      op = _ref.op,
      val = _ref.val,
      dim = _ref.dim;
    var dimPlural = dim.charAt(0).toUpperCase() + dim.slice(1) + "s";
    var method = op + dimPlural;
    return "".concat(date, ".").concat(method, "(").concat(val, ")");
  },
  spelImport: function spelImport(spel) {
    var _spel$methodName;
    var date, op, val, dim;
    var matchRes = (_spel$methodName = spel.methodName) === null || _spel$methodName === void 0 ? void 0 : _spel$methodName.match(/^(minus|plus)(\w+)s$/);
    if (matchRes) {
      dim = matchRes[2].toLowerCase();
      op = matchRes[1];
      if (["minus", "plus"].includes(op)) {
        if (["day", "week", "month", "year"].includes(dim)) {
          op = {
            type: "string",
            val: op
          };
          dim = {
            type: "string",
            val: dim
          };
          val = spel.args[0];
          date = spel.obj;
          return {
            date: date,
            op: op,
            val: val,
            dim: dim
          };
        }
      }
    }
  },
  jsonLogic: function jsonLogic(_ref2) {
    var date = _ref2.date,
      op = _ref2.op,
      val = _ref2.val,
      dim = _ref2.dim;
    return {
      "date_add": [date, val * (op == "minus" ? -1 : +1), dim]
    };
  },
  jsonLogicImport: function jsonLogicImport(v) {
    var date = v["date_add"][0];
    var val = Math.abs(v["date_add"][1]);
    var op = v["date_add"][1] >= 0 ? "plus" : "minus";
    var dim = v["date_add"][2];
    return [date, op, val, dim];
  },
  jsonLogicCustomOps: {
    date_add: {}
  },
  // MySQL
  //todo: other SQL dialects?
  sqlFormatFunc: function sqlFormatFunc(_ref3) {
    var date = _ref3.date,
      op = _ref3.op,
      val = _ref3.val,
      dim = _ref3.dim;
    return "DATE_ADD(".concat(date, ", INTERVAL ").concat(parseInt(val) * (op == "minus" ? -1 : +1), " ").concat(dim.replace(/^'|'$/g, ""), ")");
  },
  mongoFormatFunc: null,
  //todo: support?
  formatFunc: function formatFunc(_ref4) {
    var date = _ref4.date,
      op = _ref4.op,
      val = _ref4.val,
      dim = _ref4.dim;
    return !val ? date : "".concat(date, " ").concat(op == "minus" ? "-" : "+", " ").concat(val, " ").concat(dim);
  },
  args: {
    date: {
      label: "Date",
      type: "datetime",
      defaultValue: {
        func: "NOW",
        args: []
      },
      valueSources: ["func", "field", "value"],
      spelEscapeForFormat: true
    },
    op: {
      label: "Op",
      type: "select",
      defaultValue: "plus",
      valueSources: ["value"],
      mainWidgetProps: {
        customProps: {
          showSearch: false
        }
      },
      fieldSettings: {
        listValues: {
          plus: "+",
          minus: "-"
        }
      },
      spelEscapeForFormat: false
    },
    val: {
      label: "Value",
      type: "number",
      fieldSettings: {
        min: 0
      },
      defaultValue: 0,
      valueSources: ["value"],
      spelEscapeForFormat: false
    },
    dim: {
      label: "Dimension",
      type: "select",
      defaultValue: "day",
      valueSources: ["value"],
      mainWidgetProps: {
        customProps: {
          showSearch: false
        }
      },
      fieldSettings: {
        listValues: {
          day: "day",
          week: "week",
          month: "month",
          year: "year"
        }
      },
      spelEscapeForFormat: false
    }
  }
};
var LOWER = exports.LOWER = {
  label: "Lowercase",
  mongoFunc: "$toLower",
  jsonLogic: "toLowerCase",
  spelFunc: "${str}.toLowerCase()",
  //jsonLogicIsMethod: true, // Removed in JsonLogic 2.x due to Prototype Pollution
  jsonLogicCustomOps: {
    toLowerCase: {}
  },
  returnType: "text",
  args: {
    str: {
      label: "String",
      type: "text",
      valueSources: ["value", "field", "func"]
    }
  }
};
var UPPER = exports.UPPER = {
  label: "Uppercase",
  mongoFunc: "$toUpper",
  jsonLogic: "toUpperCase",
  spelFunc: "${str}.toUpperCase()",
  //jsonLogicIsMethod: true, // Removed in JsonLogic 2.x due to Prototype Pollution
  jsonLogicCustomOps: {
    toUpperCase: {}
  },
  returnType: "text",
  args: {
    str: {
      label: "String",
      type: "text",
      valueSources: ["value", "field", "func"]
    }
  }
};
var LINEAR_REGRESSION = exports.LINEAR_REGRESSION = {
  label: "Linear regression",
  returnType: "number",
  formatFunc: function formatFunc(_ref5, _) {
    var coef = _ref5.coef,
      bias = _ref5.bias,
      val = _ref5.val;
    return "(".concat(coef, " * ").concat(val, " + ").concat(bias, ")");
  },
  sqlFormatFunc: function sqlFormatFunc(_ref6) {
    var coef = _ref6.coef,
      bias = _ref6.bias,
      val = _ref6.val;
    return "(".concat(coef, " * ").concat(val, " + ").concat(bias, ")");
  },
  spelFormatFunc: function spelFormatFunc(_ref7) {
    var coef = _ref7.coef,
      bias = _ref7.bias,
      val = _ref7.val;
    return "(".concat(coef, " * ").concat(val, " + ").concat(bias, ")");
  },
  spelImport: function spelImport(spel) {
    var coef, val, bias, a;
    if (spel.type === "op-plus") {
      var _spel$children = (0, _slicedToArray2["default"])(spel.children, 2);
      a = _spel$children[0];
      bias = _spel$children[1];
      if (a.type === "op-multiply") {
        var _a$children = (0, _slicedToArray2["default"])(a.children, 2);
        coef = _a$children[0];
        val = _a$children[1];
        return {
          coef: coef,
          val: val,
          bias: bias
        };
      }
    }
  },
  mongoFormatFunc: function mongoFormatFunc(_ref8) {
    var coef = _ref8.coef,
      bias = _ref8.bias,
      val = _ref8.val;
    return {
      "$sum": [{
        "$multiply": [coef, val]
      }, bias]
    };
  },
  jsonLogic: function jsonLogic(_ref9) {
    var coef = _ref9.coef,
      bias = _ref9.bias,
      val = _ref9.val;
    return {
      "+": [{
        "*": [coef, val]
      }, bias]
    };
  },
  jsonLogicImport: function jsonLogicImport(v) {
    var coef = v["+"][0]["*"][0];
    var val = v["+"][0]["*"][1];
    var bias = v["+"][1];
    return [coef, val, bias];
  },
  renderBrackets: ["", ""],
  renderSeps: [" * ", " + "],
  args: {
    coef: {
      label: "Coef",
      type: "number",
      defaultValue: 1,
      valueSources: ["value"]
    },
    val: {
      label: "Value",
      type: "number",
      valueSources: ["value", "field"]
    },
    bias: {
      label: "Bias",
      type: "number",
      defaultValue: 0,
      valueSources: ["value"]
    }
  }
};