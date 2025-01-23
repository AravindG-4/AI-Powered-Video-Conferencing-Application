const webpack = require('webpack');

module.exports = {
  resolve: {
    fallback: {
      process: require.resolve("process/browser"),
    },
  },
  plugins: [
    new webpack.DefinePlugin({
      "process.env.NODE_ENV": JSON.stringify("development"), // or "production"
    }),
  ],  
};
