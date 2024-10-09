"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");
var _typeof = require("@babel/runtime/helpers/typeof");
Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;
var _classCallCheck2 = _interopRequireDefault(require("@babel/runtime/helpers/classCallCheck"));
var _createClass2 = _interopRequireDefault(require("@babel/runtime/helpers/createClass"));
var _assertThisInitialized2 = _interopRequireDefault(require("@babel/runtime/helpers/assertThisInitialized"));
var _inherits2 = _interopRequireDefault(require("@babel/runtime/helpers/inherits"));
var _possibleConstructorReturn2 = _interopRequireDefault(require("@babel/runtime/helpers/possibleConstructorReturn"));
var _getPrototypeOf2 = _interopRequireDefault(require("@babel/runtime/helpers/getPrototypeOf"));
var _react = _interopRequireWildcard(require("react"));
var _propTypes = _interopRequireDefault(require("prop-types"));
var _tree = _interopRequireDefault(require("../stores/tree"));
var _context = _interopRequireDefault(require("../stores/context"));
var _redux = require("redux");
var _reactRedux = require("react-redux");
var actions = _interopRequireWildcard(require("../actions"));
var _configUtils = require("../utils/configUtils");
var _stuff = require("../utils/stuff");
var _defaultUtils = require("../utils/defaultUtils");
var _validation = require("../utils/validation");
var _reactUtils = require("../utils/reactUtils");
var _Query = _interopRequireDefault(require("./Query"));
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function _getRequireWildcardCache(nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || _typeof(obj) !== "object" && typeof obj !== "function") { return { "default": obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj["default"] = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
function _createSuper(Derived) { var hasNativeReflectConstruct = _isNativeReflectConstruct(); return function _createSuperInternal() { var Super = (0, _getPrototypeOf2["default"])(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = (0, _getPrototypeOf2["default"])(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return (0, _possibleConstructorReturn2["default"])(this, result); }; }
function _isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
var QueryContainer = /*#__PURE__*/function (_Component) {
  (0, _inherits2["default"])(QueryContainer, _Component);
  var _super = _createSuper(QueryContainer);
  function QueryContainer(props, context) {
    var _this;
    (0, _classCallCheck2["default"])(this, QueryContainer);
    _this = _super.call(this, props, context);
    _this.shouldComponentUpdate = (0, _reactUtils.liteShouldComponentUpdate)((0, _assertThisInitialized2["default"])(_this), {
      value: function value(nextValue, prevValue, state) {
        return false;
      }
    });
    (0, _reactUtils.useOnPropsChanged)((0, _assertThisInitialized2["default"])(_this));
    _this.getMemoizedConfig = (0, _configUtils.createConfigMemo)();
    _this.getMemoizedTree = (0, _validation.createValidationMemo)();
    var config = _this.getMemoizedConfig(props);
    var tree = props.value;
    var validatedTree = _this.getMemoizedTree(config, tree);
    var reducer = (0, _tree["default"])(config, validatedTree, _this.getMemoizedTree);
    var store = (0, _redux.createStore)(reducer);
    _this.state = {
      store: store,
      config: config
    };
    return _this;
  }
  (0, _createClass2["default"])(QueryContainer, [{
    key: "onPropsChanged",
    value: function onPropsChanged(nextProps) {
      var _this2 = this;
      // compare configs
      var oldConfig = this.state.config;
      var nextConfig = this.getMemoizedConfig(nextProps);
      var isConfigChanged = oldConfig !== nextConfig;

      // compare trees
      var storeValue = this.state.store.getState().tree;
      var isTreeChanged = !(0, _stuff.immutableEqual)(nextProps.value, this.props.value) && !(0, _stuff.immutableEqual)(nextProps.value, storeValue);
      var currentTree = isTreeChanged ? nextProps.value || (0, _defaultUtils.defaultRoot)(nextProps) : storeValue;
      if (isConfigChanged) {
        this.setState({
          config: nextConfig
        });
      }
      if (isTreeChanged || isConfigChanged) {
        var validatedTree = this.getMemoizedTree(nextConfig, currentTree, oldConfig);
        return Promise.resolve().then(function () {
          _this2.state.store.dispatch(actions.tree.setTree(nextConfig, validatedTree));
        });
      }
    }
  }, {
    key: "render",
    value: function render() {
      // `get_children` is deprecated!
      var _this$props = this.props,
        renderBuilder = _this$props.renderBuilder,
        get_children = _this$props.get_children,
        onChange = _this$props.onChange,
        settings = _this$props.settings;
      var _this$state = this.state,
        config = _this$state.config,
        store = _this$state.store;
      var QueryWrapper = settings.renderProvider;
      return /*#__PURE__*/_react["default"].createElement(QueryWrapper, {
        config: config
      }, /*#__PURE__*/_react["default"].createElement(_reactRedux.Provider, {
        store: store,
        context: _context["default"]
      }, /*#__PURE__*/_react["default"].createElement(_Query["default"], {
        config: config,
        getMemoizedTree: this.getMemoizedTree,
        onChange: onChange,
        renderBuilder: renderBuilder || get_children
      })));
    }
  }]);
  return QueryContainer;
}(_react.Component);
exports["default"] = QueryContainer;
QueryContainer.propTypes = {
  //config
  conjunctions: _propTypes["default"].object.isRequired,
  fields: _propTypes["default"].object.isRequired,
  types: _propTypes["default"].object.isRequired,
  operators: _propTypes["default"].object.isRequired,
  widgets: _propTypes["default"].object.isRequired,
  settings: _propTypes["default"].object.isRequired,
  onChange: _propTypes["default"].func,
  renderBuilder: _propTypes["default"].func,
  value: _propTypes["default"].any //instanceOf(Immutable.Map)
};