import * as React from 'react'
import { View, Text, StyleSheet } from 'react-native'
import {
  GestureHandlerRootView,
  TouchableOpacity,
} from 'react-native-gesture-handler'
import { enableScreens } from 'react-native-screens'
import { NavigationContainer } from '@react-navigation/native'
import { createNativeStackNavigator } from '@react-navigation/native-stack'
import Transcribe from './Transcribe'
import TranscribeData from './TranscribeData'
import Vad from './Vad'
import Bench from './Bench'

enableScreens()

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  button: {
    margin: 10,
    padding: 10,
    backgroundColor: '#333',
    borderRadius: 5,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
  },
})

function HomeScreen({ navigation }: { navigation: any }) {
  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('Transcribe')}
      >
        <Text style={styles.buttonText}>Example: Transcribe File / Realtime</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('TranscribeData')}
      >
        <Text style={styles.buttonText}>Example: Transcribe Data</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('Vad')}
      >
        <Text style={styles.buttonText}>Example: VAD</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('Bench')}
      >
        <Text style={styles.buttonText}>Benchmark</Text>
      </TouchableOpacity>
    </View>
  )
}

const Stack = createNativeStackNavigator()

function App() {
  return (
    <GestureHandlerRootView>
      <NavigationContainer>
        <Stack.Navigator>
          <Stack.Screen name="Home" component={HomeScreen} />
          <Stack.Screen name="Transcribe" component={Transcribe} />
          <Stack.Screen name="TranscribeData" component={TranscribeData} />
          <Stack.Screen name="Vad" component={Vad} />
          <Stack.Screen name="Bench" component={Bench} />
        </Stack.Navigator>
      </NavigationContainer>
    </GestureHandlerRootView>
  )
}

export default App
