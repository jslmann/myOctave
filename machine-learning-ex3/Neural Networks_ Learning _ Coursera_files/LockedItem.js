"use strict";define("bundles/locking/components/LockedItem",["require","exports","module","q","react-with-addons","bundles/ondemand/utils/componentDataFetcher","bundles/phoenix/components/LockedByItem","pages/open-course/common/singletons/courseMaterials","css!./__styles__/LockedItem"],function(require,exports,module){var t=require("q"),e=require("react-with-addons"),s=require("bundles/ondemand/utils/componentDataFetcher"),a=require("bundles/phoenix/components/LockedByItem"),o=require("pages/open-course/common/singletons/courseMaterials");require("css!./__styles__/LockedItem");var n=e.createClass({displayName:"LockedItem",propTypes:{itemMetadata:e.PropTypes.object.isRequired,courseMaterials:e.PropTypes.object.isRequired},render:function render(){var t=this.props,s=t.itemMetadata,o=t.courseMaterials,n=o.getItemMetadata(s.get("lockableByItem"));return e.createElement("div",{className:"c-open-single-page c-locked-item"},e.createElement("div",{className:"bt3-row"},e.createElement("div",{className:"bt3-col-md-12"},e.createElement(a,{itemMetadata:s,previousItemMetadata:n}))))}}),c=s(n,function(e,s){return t.all([o]).spread(function(e){return{courseMaterials:e}})});module.exports=c});