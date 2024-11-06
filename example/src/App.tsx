import * as React from 'react'
import { View, Text, Pressable, StyleSheet } from 'react-native'
import { NavigationContainer } from '@react-navigation/native'
import { createNativeStackNavigator } from '@react-navigation/native-stack'
import Transcribe from './Transcribe'
import Bench from './Bench'

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
      <Pressable
        style={styles.button}
        onPress={() => navigation.navigate('Transcribe')}
      >
        <Text style={styles.buttonText}>Transcribe Examples</Text>
      </Pressable>
      <Pressable
        style={styles.button}
        onPress={() => navigation.navigate('Bench')}
      >
        <Text style={styles.buttonText}>Benchmark</Text>
      </Pressable>
    </View>
  )
}

const Stack = createNativeStackNavigator()

function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Transcribe" component={Transcribe} />
        <Stack.Screen name="Bench" component={Bench} />
      </Stack.Navigator>
    </NavigationContainer>
  )
}

export default App
