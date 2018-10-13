var bmap = require('../../libs/bmap-wx/bmap-wx.min.js');
var nowtoday=require('../../utils/util.js')
var wxMarkerData = [];	//定位成功回调对象
Page({
  data: {
    // 初始化数据，可设置为空
    detail:{},
    weather: {
        temperature: 18
              },
        forecast: [
          {
            date: '18日星期五',
            weather: '阴',
            temperature: '高温 16℃',
            wind: '南风',
          
          }, {
            date: '18日星期五',
            weather: '阴',
            temperature: '高温 16℃',
            wind: '微风级'
          }, {
            date: '18日星期五',
            weather: '阴',
            temperature: '高温 16℃',
            wind: '微风级'
          }, {
            date: '18日星期五',
            weather: '阴',
            temperature: '高温 16℃',
            wind: '微风级'
          }
        ],
    today: '2018-6-7',
    city: '北京',    //城市名称
    inputCity: '', //输入查询的城市名称
    // ak,百度地图api申请
    ak: "5UlOO40DmXUVLBOAjQkeNiyTdtR3VXKm",	//填写申请到的ak
    markers: [],
    // 经度，下行纬度
    longitude: '',	
    latitude: '',	
    // 地址信息，较详细
    address: '',
    	// 行政区域信息
    cityInfo: {}	
  },

  onLoad: function (options) {
    // 主要功能：获得地理位置，调用搜索函数（下面定义）完成本地天气的显示
    this.setData({
      today:nowtoday.formatTime(new Date()).split(' ')[0]
    })
    var that = this;
    
    // 新建bmap对象 ，必须参数ak
    var BMap = new bmap.BMapWX({
      ak: that.data.ak
    });
    // 获取失败的处理
    var fail = function (data) {
      console.log(data);
    };
    // 获取成功的处理
    var success = function (data) {
      //返回数据内，已经包含经纬度
      console.log(data);
      // 格式化数据
      wxMarkerData = data.wxMarkerData;
      //数据赋值，具体看console中返回的数据详细信息
      that.setData({
        markers: wxMarkerData,
        latitude: wxMarkerData[0].latitude,
        longitude: wxMarkerData[0].longitude,
        address: wxMarkerData[0].address,
        cityInfo: data.originalData.result.addressComponent,
        // 下行，有的api不支持输入地址中出现xx市，只要地名
        city: data.originalData.result.addressComponent.city.replace('市','')
      });
    }
    // 百度地图api，解码功能，
    BMap.regeocoding({
      fail: fail,
      success: success
    });
    // 调用searchWeather函数，传入当前城市名称
    that.searchWeather(that.data.city)
    // 注释部分为调用百度自带的api来显示本地天气情况
    // var that = this;
    // // // 新建bmap对象 
    // var BMap1 = new bmap.BMapWX({
    //   ak: that.data.ak
    // });
    // // regeocoding检索请求 
  
    // var fail1 = function (data) {
    //   console.log(data);
    // };
    // var success1 = function (data) {
    //   
    //   console.log(data);
    
    //   that.setData({
    //     weather: data.currentWeather[0],
    //     forecast: data.originalData.results[0].weather_data
    //   });
    // } 
    //   BMap.weather({
    //     fail: fail1,
    //     success: success1
    //   });
},
  searchWeather: function (cityName) {
    // 调用百度地图天气api获取指定城市的天气
    var self = this;
    wx.request({
      url: 'https://api.map.baidu.com/telematics/v3/weather?location='+cityName+'&output=json&ak='+self.data.ak,
      
      header: { 'Content-Type': 'application/json' },
      success: function (res) {
        console.log(res.data.results)

        self.setData({
          weather:res.data.results[0].weather_data[0],
          forecast:res.data.results[0].weather_data,
          city:res.data.results[0].currentCity,
          suggestion:res.data.results[0].index[0].des,
          pm25:res.data.results[0].pm25
        })
        
      },
      fail:function(res){
        console.log(res)
      }
    })
  },
  inputing: function (e) {
    // 获取输入框内容，写入inputCity
    this.setData({
      inputCity: e.detail.value
    });
  },
  bindSearch: function () {
    // 点击搜索时执行搜索天气
    this.searchWeather(this.data.inputCity);
  } 

})